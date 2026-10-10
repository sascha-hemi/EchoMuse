package outchain

import "math"

// mbclBand is one band of a multi-band compressor-limiter as the device's
// MBCL.cfg describes it: a compressor (input gain, ratio, threshold, a floor
// on how far it may pull down) followed by a peak limiter.
type mbclBand struct {
	CompInVol   float64 `json:"comp_inVol"`
	CompRatio   float64 `json:"comp_ratio"`
	CompThresh  float64 `json:"comp_thresh"`
	CompGainMin float64 `json:"comp_gainMin"`
	LimInVol    float64 `json:"lim_inVol"`
	LimThresh   float64 `json:"lim_thresh"`
	LimRelease  float64 `json:"lim_release"`
}

type mbclLimiterSpec struct {
	LimInVol   float64 `json:"lim_inVol"`
	LimThresh  float64 `json:"lim_thresh"`
	LimRelease float64 `json:"lim_release"`
}

// mbclSpec is MBCL.cfg: crossover frequencies, four bands, a full-band
// limiter, and an input gain ahead of all of it.
type mbclSpec struct {
	Bypass          bool            `json:"Bypass"`
	PreFilterBypass bool            `json:"PreFilterBypass"`
	InVol           float64         `json:"inVol"`
	NumBands        int             `json:"NumBands"`
	FC              []float64       `json:"FilterBank FC"`
	Bands           []mbclBand      `json:"Bands Definition"`
	FullBand        mbclLimiterSpec `json:"Full-band limiter"`
}

// The file gives no compressor time constants. These are an assumption —
// typical program-material values — and the one part of this stage that is
// not read from the device.
const (
	mbclCompAttackMs  = 5.0
	mbclCompReleaseMs = 100.0
)

// lr4 is a 4th-order Linkwitz-Riley filter: two 2nd-order Butterworths in
// cascade. LP and HP of the same frequency sum to an allpass.
type lr4 struct{ a, b biquad }

func newLR4(fc, fs float64, high bool) lr4 {
	s := butter2(fc, fs, high)
	return lr4{s, s}
}

func (f *lr4) step(x float64) float64 { return f.b.step(f.a.step(x)) }
func (f *lr4) reset()                 { f.a.reset(); f.b.reset() }

// lr4ap is the allpass an LR4 crossover at fc imposes on the other branches,
// as LP+HP, so every band reaches the sum with the same phase.
type lr4ap struct{ lp, hp lr4 }

func newLR4AP(fc, fs float64) lr4ap {
	return lr4ap{newLR4(fc, fs, false), newLR4(fc, fs, true)}
}
func (f *lr4ap) step(x float64) float64 { return f.lp.step(x) + f.hp.step(x) }
func (f *lr4ap) reset()                 { f.lp.reset(); f.hp.reset() }

// dynamics is one compressor-plus-limiter, in the chain's S16 signal domain:
// thresholds in dBFS against fullScale.
type dynamics struct {
	compIn, compThr, compSlope, compFloor float64 // linear in, dBFS, 1-1/ratio, max reduction dB
	compAtt, compRel                      float64 // one-pole coefficients
	compEnvDb                             float64 // smoothed gain reduction, dB (<= 0)
	compOn                                bool

	limIn, limThr, limRel float64
	limEnv                float64
}

func newDynamics(b mbclBand, fs float64) dynamics {
	d := dynamics{
		compIn:  dbToGain(b.CompInVol),
		compThr: b.CompThresh,
		compOn:  b.CompRatio > 1,
		limIn:   dbToGain(b.LimInVol),
		limThr:  fullScale * dbToGain(b.LimThresh),
		limRel:  onePole(b.LimRelease, fs),
		compAtt: onePole(mbclCompAttackMs, fs),
		compRel: onePole(mbclCompReleaseMs, fs),
	}
	if d.compOn {
		d.compSlope = 1 - 1/b.CompRatio
	}
	d.compFloor = math.Min(b.CompGainMin, 0)
	return d
}

// onePole is the smoothing coefficient for a time constant in ms.
func onePole(ms, fs float64) float64 {
	if ms <= 0 {
		return 0
	}
	return math.Exp(-1 / (ms / 1000 * fs))
}

func (d *dynamics) step(x float64) float64 {
	x *= d.compIn
	if d.compOn {
		a := math.Abs(x)
		target := 0.0
		if a > 0 {
			over := 20*math.Log10(a/fullScale) - d.compThr
			if over > 0 {
				target = -over * d.compSlope
				if target < d.compFloor {
					target = d.compFloor
				}
			}
		}
		k := d.compRel
		if target < d.compEnvDb {
			k = d.compAtt
		}
		d.compEnvDb = target + k*(d.compEnvDb-target)
		x *= dbToGain(d.compEnvDb)
	}
	return limitPeak(x*d.limIn, d.limThr, d.limRel, &d.limEnv)
}

// limitPeak is an instant-attack, exponential-release peak limiter: the
// envelope jumps to a new peak and decays from it, and the gain holds the
// output at the threshold. No look-ahead, so no added latency.
func limitPeak(x, thr, rel float64, env *float64) float64 {
	a := math.Abs(x)
	if a > *env {
		*env = a
	} else {
		*env *= rel
	}
	if *env > thr {
		return x * thr / *env
	}
	return x
}

func (d *dynamics) reset() { d.compEnvDb, d.limEnv = 0, 0 }

// mbcl is the multi-band compressor-limiter: input gain, a 4-band LR4
// crossover with allpass phase alignment, per-band dynamics, the sum, and a
// full-band limiter.
type mbcl struct {
	inVol  float64
	lp, hp [3]lr4
	ap     [3]lr4ap // band 1 through fc2 and fc3, band 2 through fc3
	bands  [4]dynamics
	full   struct {
		in, thr, rel, env float64
	}
}

func newMBCL(s *mbclSpec, fs float64) *mbcl {
	m := &mbcl{inVol: dbToGain(s.InVol)}
	for i := 0; i < 3; i++ {
		m.lp[i] = newLR4(s.FC[i], fs, false)
		m.hp[i] = newLR4(s.FC[i], fs, true)
	}
	m.ap[0] = newLR4AP(s.FC[1], fs) // band 1, compensating fc2
	m.ap[1] = newLR4AP(s.FC[2], fs) // band 1, compensating fc3
	m.ap[2] = newLR4AP(s.FC[2], fs) // band 2, compensating fc3
	for i := 0; i < 4; i++ {
		m.bands[i] = newDynamics(s.Bands[i], fs)
	}
	m.full.in = dbToGain(s.FullBand.LimInVol)
	m.full.thr = fullScale * dbToGain(s.FullBand.LimThresh)
	m.full.rel = onePole(s.FullBand.LimRelease, fs)
	return m
}

func (m *mbcl) step(x float64) float64 {
	x *= m.inVol
	b1 := m.lp[0].step(x)
	r := m.hp[0].step(x)
	b2 := m.lp[1].step(r)
	r = m.hp[1].step(r)
	b3 := m.lp[2].step(r)
	b4 := m.hp[2].step(r)
	b1 = m.ap[1].step(m.ap[0].step(b1))
	b2 = m.ap[2].step(b2)
	y := m.bands[0].step(b1) + m.bands[1].step(b2) + m.bands[2].step(b3) + m.bands[3].step(b4)
	return limitPeak(y*m.full.in, m.full.thr, m.full.rel, &m.full.env)
}

func (m *mbcl) reset() {
	for i := range m.lp {
		m.lp[i].reset()
		m.hp[i].reset()
		m.ap[i].reset()
	}
	for i := range m.bands {
		m.bands[i].reset()
	}
	m.full.env = 0
}
