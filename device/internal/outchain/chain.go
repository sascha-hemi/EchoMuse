package outchain

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
)

// Params is the chain's whole configuration. Defaults match
// em_db.DEFAULT_DEVICE_CONFIG.
type Params struct {
	Bands              [NumBands]float64
	Loudness           bool
	GuardEnabled       bool
	GuardDb            float64
	LimiterEnabled     bool
	LimiterThresholdDb float64
	LimiterReleaseMs   float64
	// SpeakerTuning allows the device's speaker tuning stage (tuning.go) to
	// run when one is loaded. Not a controller setting: the device clears it
	// while a plug is in the jack, because the tuning is for the internal
	// speaker.
	SpeakerTuning bool
}

// DefaultParams mirrors the controller's defaults, so a device that has not
// yet had a config push sounds the way the controller would have made it.
func DefaultParams() Params {
	return Params{
		GuardEnabled:       true,
		GuardDb:            -30,
		LimiterEnabled:     true,
		LimiterThresholdDb: -1,
		LimiterReleaseMs:   150,
		SpeakerTuning:      true,
	}
}

// String is the one-line description em_eq.describe_chain gives, so device
// and controller logs read the same way.
func (p Params) String() string {
	eqs := "flat"
	if !isFlat(p.Bands, false) {
		parts := make([]string, NumBands)
		for i, b := range p.Bands {
			if b == 0 {
				parts[i] = "0"
			} else {
				parts[i] = fmt.Sprintf("%+g", b)
			}
		}
		eqs = strings.Join(parts, "/")
	}
	boost := "off"
	if p.Loudness {
		boost = "on"
	}
	guard := "off"
	if p.GuardEnabled {
		guard = fmt.Sprintf("%gdB", math.Min(p.GuardDb, 0))
	}
	lim := "off"
	if p.LimiterEnabled {
		lim = fmt.Sprintf("%gdB/%gms", math.Min(p.LimiterThresholdDb, 0), p.LimiterReleaseMs)
	}
	return fmt.Sprintf("eq=%s speech_boost=%s guard=%s limiter=%s", eqs, boost, guard, lim)
}

// Chain runs EQ → bass guard → limiter on stereo S16_LE periods.
//
// Order is em_eq's: the guard removes excursion the driver cannot deliver,
// THEN the limiter catches what is left. Limiting first would spend gain
// reduction on bass about to be thrown away.
//
// Dual-mono audio — everything the controller sends, since its wire is mono
// and toStereo duplicates it — is processed ONCE: L and R are averaged, which
// is exact when they are equal, and written back to both. Processing two
// identical channels would double the cost for nothing.
//
// Stereo audio (Sendspin with a plug in the jack, #273) is processed per
// channel: EQ on each, and the bass guard and limiter LINKED, one gain from
// the louder side applied to both, so limiting never moves the image. The
// chain switches on the first period whose channels differ, seeding the
// right channel's state from the left's, which is what it would have held on
// the dual-mono audio before; it goes back to mono processing only when it
// resets on silence. With L == R the stereo path computes exactly what the
// mono one does, so the switch itself is inaudible.
//
// Process runs on the ALSA write goroutine only. SetParams and SetActive may
// be called from anywhere; they take effect at the next period.
type Chain struct {
	fs float64

	active atomic.Bool // false: Process is a passthrough

	mu      sync.Mutex
	pending *Params // set by SetParams, taken by Process

	// Owned by the ALSA goroutine.
	params  Params
	eq      eq
	eqR     eq // the right channel's, in stereo
	guard   *bassGuard
	lim     *limiter
	idle    bool // state is all zero and input is silence
	running bool // active on the previous period
	stereo  bool // processing L and R apart, since a period differed

	// The speaker tuning stage, when the device has one (SetTuning), and the
	// volume that picks its FIR (SetVolumePercent).
	tuningNext atomic.Pointer[tuning]
	tuning     *tuning
	volumePct  atomic.Int32
}

// New builds a chain at the given sample rate, inactive, with DefaultParams.
func New(sampleRate int) *Chain {
	fs := float64(sampleRate)
	c := &Chain{
		fs:    fs,
		eq:    eq{fs: fs},
		eqR:   eq{fs: fs},
		guard: newBassGuard(fs),
		lim:   newLimiter(fs),
		idle:  true,
	}
	c.volumePct.Store(50)
	c.apply(DefaultParams())
	return c
}

// SetTuning installs the speaker tuning stage. It runs ahead of the EQ, on mono
// audio only, while Params.SpeakerTuning is set; while it runs, the bass
// guard is skipped, because the tuning's own multi-band compressor is what
// protects the driver from the bass the tuning adds. A nil or empty spec
// removes it. Takes effect at the next period.
func (c *Chain) SetTuning(s *TuningSpec) {
	if s.Empty() {
		c.tuningNext.Store(&tuning{})
		return
	}
	c.tuningNext.Store(newTuning(s, c.fs))
}

// SetVolumePercent tells the tuning stage the device volume, 0-100, which
// picks its FIR.
func (c *Chain) SetVolumePercent(pct int) {
	if pct < 0 {
		pct = 0
	} else if pct > 100 {
		pct = 100
	}
	c.volumePct.Store(int32(pct))
}

// tuned runs the tuning stage over a mono block when it applies, and returns
// nil when it does not (no tuning, disabled, or stereo).
func (c *Chain) tuned(n int, sample func(i int) float64) []float64 {
	t := c.tuning
	if t == nil || (t.fir == nil && t.mbcl == nil && len(t.peq) == 0) || !c.params.SpeakerTuning || c.stereo {
		return nil
	}
	blk := t.block(n)
	for i := range blk {
		blk[i] = sample(i)
	}
	t.process(blk, int(c.volumePct.Load()), 1)
	return blk
}

// SetActive turns processing on or off. Off is a passthrough, which is what
// a device must do while its controller is still processing the audio itself:
// the chain applied twice is a doubled EQ curve and a second limiter.
func (c *Chain) SetActive(on bool) { c.active.Store(on) }

// Active reports whether the chain is processing.
func (c *Chain) Active() bool { return c.active.Load() }

// SetParams queues a new configuration for the next period. Everything is
// updated in place — filter states, the limiter's delay line and its gain
// carry across — so a change mid-song does not click.
func (c *Chain) SetParams(p Params) {
	c.mu.Lock()
	c.pending = &p
	c.mu.Unlock()
}

func (c *Chain) apply(p Params) {
	c.params = p
	c.eq.set(p.Bands, p.Loudness)
	c.eqR.set(p.Bands, p.Loudness)
	c.guard.enabled = p.GuardEnabled
	c.guard.floorDb = math.Min(p.GuardDb, 0)
	c.lim.enabled = p.LimiterEnabled
	c.lim.setParams(p.LimiterThresholdDb, p.LimiterReleaseMs, c.fs)
}

// takePending applies a queued SetParams. Returns the params that are now in
// force and whether they changed.
func (c *Chain) takePending() (Params, bool) {
	c.mu.Lock()
	p := c.pending
	c.pending = nil
	c.mu.Unlock()
	if p == nil {
		return c.params, false
	}
	changed := *p != c.params
	c.apply(*p)
	return c.params, changed
}

// Idle reports whether processing a silent period would return silence
// unchanged, so the caller can skip it. True while inactive.
func (c *Chain) Idle() bool {
	return !c.active.Load() || c.idle
}

// Process runs the chain over one stereo S16_LE period, IN PLACE, and returns
// the same buffer. The caller must own buf — never pass a shared silence
// buffer.
//
// The first return is non-nil only when a queued SetParams changed the chain
// on this period, so the caller can log what the audio is now going through.
func (c *Chain) Process(buf []byte) (applied *Params) {
	return c.process(buf, nil)
}

// ProcessAtLevel is Process with the playback volume applied FIRST, ramped
// from g0 to g1 across the period the way the speaker's volume stage ramps,
// so everything after it runs at the level the speaker gets. The caller then
// applies no volume of its own.
//
// It exists for the speaker tuning: the tuning lifts the bass by up to ~20dB
// and its compressor keeps that within what the driver takes AT THE PLAYBACK
// LEVEL. Processed at full scale with the volume after, the lifted bass sat
// far above full scale, and the limiter pulled the whole signal down on every
// bass note — the treble pumped with the bass (Radar1, 2026-10-10).
func (c *Chain) ProcessAtLevel(buf []byte, g0, g1 float64) (applied *Params) {
	frames := len(buf) / 4
	if frames == 0 {
		return c.process(buf, nil)
	}
	step := (g1 - g0) / float64(frames)
	return c.process(buf, func(i int) float64 { return g0 + step*float64(i+1) })
}

// TuningOn reports whether the speaker tuning would run on mono audio now.
// ALSA goroutine only, like Process.
func (c *Chain) TuningOn() bool {
	t := c.tuning
	if n := c.tuningNext.Load(); n != nil {
		t = n
	}
	return t != nil && (t.fir != nil || t.mbcl != nil || len(t.peq) > 0) &&
		c.params.SpeakerTuning && c.active.Load()
}

func (c *Chain) process(buf []byte, gain func(i int) float64) (applied *Params) {
	applied, active := c.beginProcess()
	if !active {
		if gain != nil {
			scaleS16(buf, gain)
		}
		return applied
	}
	g := gain
	if g == nil {
		g = func(int) float64 { return 1 }
	}

	if !c.stereo && !dualMono(buf) {
		c.goStereo()
	}

	frames := len(buf) / 4
	tuned := c.tuned(frames, func(i int) float64 {
		off := i * 4
		l := int16(uint16(buf[off]) | uint16(buf[off+1])<<8)
		r := int16(uint16(buf[off+2]) | uint16(buf[off+3])<<8)
		return (float64(l) + float64(r)) / 2 * g(i)
	})
	silentIn, silentOut := true, true
	for i := 0; i < frames; i++ {
		off := i * 4
		l := int16(uint16(buf[off]) | uint16(buf[off+1])<<8)
		r := int16(uint16(buf[off+2]) | uint16(buf[off+3])<<8)
		if l != 0 || r != 0 {
			silentIn = false
		}

		if c.stereo {
			gi := g(i)
			xl, xr := c.eq.step(float64(l)*gi), c.eqR.step(float64(r)*gi)
			xl, xr = c.guard.stepStereo(xl, xr)
			xl, xr = c.lim.stepStereo(xl, xr)
			sl, sr := toS16(xl), toS16(xr)
			if sl != 0 || sr != 0 {
				silentOut = false
			}
			buf[off], buf[off+1] = byte(uint16(sl)), byte(uint16(sl)>>8)
			buf[off+2], buf[off+3] = byte(uint16(sr)), byte(uint16(sr)>>8)
			continue
		}

		x := (float64(l) + float64(r)) / 2 * g(i)
		if tuned != nil {
			x = tuned[i]
		}

		x = c.eq.step(x)
		if tuned == nil {
			x = c.guard.step(x)
		}
		x = c.lim.step(x)

		s := toS16(x)
		if s != 0 {
			silentOut = false
		}
		lo, hi := byte(uint16(s)), byte(uint16(s)>>8)
		buf[off], buf[off+1], buf[off+2], buf[off+3] = lo, hi, lo, hi
	}

	// A silent period that came out silent means every filter tail has
	// decayed below one LSB. Zero the state and stop processing silence
	// until audio returns — otherwise the chain runs flat out on an idle
	// speaker, forever. The reset moves the output by less than one LSB.
	if silentIn && silentOut {
		c.reset()
		c.idle = true
	} else {
		c.idle = false
	}
	return applied
}

// ProcessFloat is Process for a wide-precision interleaved stereo period.
// fullScales contains one full-scale multiplier per frame. Samples are
// normalised into the chain's ordinary S16 signal domain and returned to the
// wide domain afterward. Keeping the chain's state normalised makes switching
// between ordinary and boosted periods seamless, while the caller retains
// headroom until the later master-volume stage.
//
// It shares the mono/stereo state with Process: a boosted response mixed over
// stereo music is processed per channel like any other stereo period, so the
// music under the reply keeps its image and the right channel's state is
// current when the next ordinary period arrives.
func (c *Chain) ProcessFloat(buf []float64, fullScales []float64) (applied *Params) {
	applied, active := c.beginProcess()
	if !active {
		return applied
	}

	if !c.stereo && !dualMonoFloat(buf) {
		c.goStereo()
	}

	frames := len(buf) / 2
	tuned := c.tuned(frames, func(i int) float64 {
		scale := 1.0
		if i < len(fullScales) && fullScales[i] > 0 {
			scale = fullScales[i]
		}
		return (buf[i*2] + buf[i*2+1]) / (2 * scale)
	})
	silentIn, silentOut := true, true
	for i := 0; i < frames; i++ {
		l, r := buf[i*2], buf[i*2+1]
		if l != 0 || r != 0 {
			silentIn = false
		}
		scale := 1.0
		if i < len(fullScales) && fullScales[i] > 0 {
			scale = fullScales[i]
		}

		if c.stereo {
			xl, xr := c.eq.step(l/scale), c.eqR.step(r/scale)
			xl, xr = c.guard.stepStereo(xl, xr)
			xl, xr = c.lim.stepStereo(xl, xr)
			xl, xr = clampS16(xl)*scale, clampS16(xr)*scale
			if math.Abs(xl) >= 1 || math.Abs(xr) >= 1 {
				silentOut = false
			}
			buf[i*2], buf[i*2+1] = xl, xr
			continue
		}

		x := (l + r) / (2 * scale)
		if tuned != nil {
			x = tuned[i]
		}
		x = c.eq.step(x)
		if tuned == nil {
			x = c.guard.step(x)
		}
		x = c.lim.step(x)
		x = clampS16(x) * scale
		if math.Abs(x) >= 1 {
			silentOut = false
		}
		buf[i*2], buf[i*2+1] = x, x
	}
	if silentIn && silentOut {
		c.reset()
		c.idle = true
	} else {
		c.idle = false
	}
	return applied
}

// goStereo switches to per-channel processing, seeding the right channel's
// state from the left's: on the dual-mono audio before, that is what it would
// have held.
func (c *Chain) goStereo() {
	c.eqR = c.eq.clone()
	c.guard.split()
	c.lim.split()
	c.stereo = true
}

func (c *Chain) beginProcess() (applied *Params, active bool) {
	if p, changed := c.takePending(); changed {
		applied = &p
	}
	if t := c.tuningNext.Swap(nil); t != nil {
		c.tuning = t
	}
	active = c.active.Load()
	if active != c.running {
		// Entering or leaving: the chain's state belongs to audio it last
		// saw, which is not the audio arriving now. Start clean.
		c.reset()
		c.running = active
	}
	return applied, active
}

// scaleS16 multiplies a stereo S16_LE period by a per-frame gain, rounding as
// the speaker's volume stage does.
func scaleS16(buf []byte, gain func(i int) float64) {
	for i := 0; i < len(buf)/4; i++ {
		g := gain(i)
		for ch := 0; ch < 2; ch++ {
			off := i*4 + ch*2
			s := math.Round(float64(int16(uint16(buf[off])|uint16(buf[off+1])<<8)) * g)
			v := toS16(s)
			buf[off], buf[off+1] = byte(uint16(v)), byte(uint16(v)>>8)
		}
	}
}

// toS16 is the backstop, then truncation toward zero —
// np.clip(...).astype(int16) in the reference.
func toS16(x float64) int16 {
	return int16(clampS16(x))
}

// clampS16 is the backstop alone, for the wide path, which keeps its
// fraction until the volume stage.
func clampS16(x float64) float64 {
	if x > ceiling {
		return ceiling
	} else if x < -fullScale {
		return -fullScale
	}
	return x
}

// dualMono reports whether every frame of a stereo S16_LE period has L == R.
func dualMono(buf []byte) bool {
	for off := 0; off+3 < len(buf); off += 4 {
		if buf[off] != buf[off+2] || buf[off+1] != buf[off+3] {
			return false
		}
	}
	return true
}

// dualMonoFloat is dualMono for an interleaved float64 period.
func dualMonoFloat(buf []float64) bool {
	for i := 0; i+1 < len(buf); i += 2 {
		if buf[i] != buf[i+1] {
			return false
		}
	}
	return true
}

func (c *Chain) reset() {
	if c.tuning != nil {
		c.tuning.reset()
	}
	c.eq.reset()
	c.eqR.reset()
	c.stereo = false
	c.guard.reset()
	c.lim.reset()
	c.idle = true
}

// Stats is the chain's instrumentation: the WORK done, as against Params,
// which is what it was set to. A stage that is on and reports 0.00dB never
// engaged, which is a different fault from one that is off.
type Stats struct {
	GuardReductionDb   float64
	LimiterReductionDb float64
	Clipped            uint64 // must stay 0 while limiting
	ClippedBypassed    uint64
}

// TakeStats returns and clears the maximum reductions since the last call.
// ALSA goroutine only.
func (c *Chain) TakeStats() Stats {
	s := Stats{
		GuardReductionDb:   c.guard.maxReductionDb,
		LimiterReductionDb: c.lim.maxReductionDb,
		Clipped:            c.lim.clipped,
		ClippedBypassed:    c.lim.clippedBypassed,
	}
	c.guard.maxReductionDb, c.lim.maxReductionDb = 0, 0
	return s
}
