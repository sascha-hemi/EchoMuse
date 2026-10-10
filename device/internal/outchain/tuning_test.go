package outchain

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The values here are made up for the tests. No vendor tuning is in the
// repository; the device reads its own at runtime.

func TestFFTRoundTrip(t *testing.T) {
	p := newFFTPlan(64)
	rng := rand.New(rand.NewSource(1))
	re, im := make([]float64, 64), make([]float64, 64)
	orig := make([]float64, 64)
	for i := range re {
		re[i] = rng.Float64()*2 - 1
		orig[i] = re[i]
	}
	p.transform(re, im, false)
	p.transform(re, im, true)
	for i := range re {
		if math.Abs(re[i]/64-orig[i]) > 1e-12 || math.Abs(im[i]/64) > 1e-12 {
			t.Fatalf("sample %d: %g%+gi, want %g", i, re[i]/64, im[i]/64, orig[i])
		}
	}
}

func directConv(h, x []float64) []float64 {
	y := make([]float64, len(x))
	for n := range x {
		s := 0.0
		for k := 0; k < len(h) && k <= n; k++ {
			s += h[k] * x[n-k]
		}
		y[n] = s
	}
	return y
}

// Overlap-save must be the linear convolution, whatever size the periods are,
// including sizes above one transform's capacity.
func TestFIRConvMatchesDirectConvolution(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	h := make([]float64, firMaxTaps)
	for i := range h {
		h[i] = (rng.Float64()*2 - 1) * math.Exp(-float64(i)/300)
	}
	x := make([]float64, 9000)
	for i := range x {
		x[i] = rng.Float64()*2 - 1
	}
	want := directConv(h, x)

	f := newFIRConv([][]float64{h})
	got := append([]float64(nil), x...)
	for off, sizes := 0, []int{2048, 1, 500, 3000, 2049, 1402}; off < len(got); {
		n := sizes[0]
		sizes = append(sizes[1:], n)
		if off+n > len(got) {
			n = len(got) - off
		}
		f.process(got[off:off+n], 0)
		off += n
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("sample %d: got %g, want %g", i, got[i], want[i])
		}
	}
}

// A filter change fades over one call and is then exactly the new filter.
func TestFIRConvCrossfadesToTheNewFilter(t *testing.T) {
	f := newFIRConv([][]float64{{1}, {0.5}})
	x := make([]float64, 1024)
	for i := range x {
		x[i] = 1000
	}
	f.process(append([]float64(nil), x...), 0)
	fade := append([]float64(nil), x...)
	f.process(fade, 1)
	if fade[0] < 990 || math.Abs(fade[len(fade)-1]-500) > 1e-9 {
		t.Fatalf("fade should run from the old to the new: first %g, last %g", fade[0], fade[len(fade)-1])
	}
	for i := 1; i < len(fade); i++ {
		if fade[i] > fade[i-1]+1e-9 {
			t.Fatalf("fade is not monotonic at %d", i)
		}
	}
	after := append([]float64(nil), x...)
	f.process(after, 1)
	for i, v := range after {
		if math.Abs(v-500) > 1e-9 {
			t.Fatalf("after the fade sample %d is %g, want 500", i, v)
		}
	}
}

func TestStripCommentsKeepsStrings(t *testing.T) {
	in := "/* head */ {\"a//b\": 1, // tail\n \"c\": \"/*x*/\" }"
	got := string(stripComments([]byte(in)))
	if !strings.Contains(got, `"a//b"`) || !strings.Contains(got, `"/*x*/"`) || strings.Contains(got, "tail") || strings.Contains(got, "head") {
		t.Fatalf("got %q", got)
	}
}

const testPEQ = `/* test */
{
    "Bypass" : false,
    "NumBiquads" : 4, // comment
    "Biquad Definitions": [
        { "FilterType": "LOW_SHELF", "Fc": 100, "Q": 0.7, "GaindB": 3.0 },
        { "FilterType": "BYPASS", "Fc": 1, "Q": 1, "GaindB": 9 },
        { "FilterType": "PEAK", "Fc": 1000, "Q": 1.0, "GaindB": 6.0 }
    ]
}`

func TestParsePEQSkipsBypassAndRejectsUnknown(t *testing.T) {
	peq, err := parsePEQ([]byte(testPEQ))
	if err != nil || len(peq) != 2 || peq[1].FilterType != "PEAK" {
		t.Fatalf("peq=%+v err=%v", peq, err)
	}
	if _, err := parsePEQ([]byte(strings.Replace(testPEQ, `"PEAK"`, `"WEIRD"`, 1))); err == nil {
		t.Fatal("accepted an unknown filter type")
	}
	if peq, err := parsePEQ([]byte(strings.Replace(testPEQ, `"Bypass" : false`, `"Bypass" : true`, 1))); err != nil || peq != nil {
		t.Fatalf("a bypassed EQ must give nothing, got %v %v", peq, err)
	}
}

func TestParseFIR(t *testing.T) {
	taps, err := parseFIR([]byte("0.5,\n-2.5e-01,\n1,\n"))
	if err != nil || len(taps) != 3 || taps[1] != -0.25 || taps[2] != 1 {
		t.Fatalf("taps=%v err=%v", taps, err)
	}
	if _, err := parseFIR([]byte(strings.Repeat("0.1,\n", firMaxTaps+1))); err == nil {
		t.Fatal("accepted more taps than the convolver holds")
	}
	if _, err := parseFIR([]byte("")); err == nil {
		t.Fatal("accepted an empty filter")
	}
}

const testMBCL = `{
    "Bypass": false, "PreFilterBypass": true, "inVol": 0, "NumBands": 4,
    "FilterBank FC": [100, 400, 4000],
    "Bands Definition": [
        {"comp_inVol":0,"comp_ratio":10,"comp_thresh":-20,"comp_gainMin":-30,"lim_inVol":0,"lim_thresh":-10,"lim_release":100},
        {"comp_inVol":0,"comp_ratio":1,"comp_thresh":0,"comp_gainMin":-30,"lim_inVol":0,"lim_thresh":0,"lim_release":50},
        {"comp_inVol":0,"comp_ratio":1,"comp_thresh":0,"comp_gainMin":-30,"lim_inVol":0,"lim_thresh":0,"lim_release":20},
        {"comp_inVol":0,"comp_ratio":1,"comp_thresh":0,"comp_gainMin":-30,"lim_inVol":0,"lim_thresh":0,"lim_release":20}
    ],
    "Full-band limiter": {"lim_inVol":0,"lim_thresh":0,"lim_release":20}
}`

func TestParseMBCLChecksTheLayout(t *testing.T) {
	if m, err := parseMBCL([]byte(testMBCL)); err != nil || m == nil || len(m.Bands) != 4 {
		t.Fatalf("m=%+v err=%v", m, err)
	}
	if _, err := parseMBCL([]byte(strings.Replace(testMBCL, "[100, 400, 4000]", "[400, 100, 4000]", 1))); err == nil {
		t.Fatal("accepted falling crossover frequencies")
	}
}

func sine(f, amp float64, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = amp * math.Sin(2*math.Pi*f*float64(i)/48000)
	}
	return x
}

func rms(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	return math.Sqrt(s / float64(len(x)))
}

// Below every threshold the crossover sums back to the input level: the
// bands are phase aligned, so nothing is lost at the split frequencies.
func TestMBCLSumsFlatBelowThresholds(t *testing.T) {
	spec, _ := parseMBCL([]byte(testMBCL))
	for _, f := range []float64{50, 100, 250, 400, 1000, 4000, 9000} {
		m := newMBCL(spec, 48000)
		x := sine(f, 100, 48000) // ~ -50 dBFS, under every threshold
		y := make([]float64, len(x))
		for i, v := range x {
			y[i] = m.step(v)
		}
		got := 20 * math.Log10(rms(y[24000:])/rms(x[24000:]))
		if math.Abs(got) > 0.5 {
			t.Errorf("%gHz: %+.2fdB through the band split, want ~0", f, got)
		}
	}
}

// Loud bass is pulled down by its band; loud treble in an uncompressed band
// is not.
func TestMBCLCompressesOnlyTheLoudBand(t *testing.T) {
	spec, _ := parseMBCL([]byte(testMBCL))
	amp := fullScale * 0.5 // -6 dBFS
	m := newMBCL(spec, 48000)
	bass := sine(40, amp, 48000)
	for i, v := range bass {
		bass[i] = m.step(v)
	}
	if peak := maxAbs(bass[24000:]); peak > fullScale*dbToGain(-10)*1.01 {
		t.Errorf("40Hz at -6dBFS peaks at %.1fdBFS, want at or under the band limiter's -10",
			20*math.Log10(peak/fullScale))
	}
	m = newMBCL(spec, 48000)
	tone := sine(1000, amp*0.5, 48000) // -12 dBFS, under the full-band 0dB limiter
	in := rms(tone[24000:])
	for i, v := range tone {
		tone[i] = m.step(v)
	}
	if d := 20 * math.Log10(rms(tone[24000:])/in); math.Abs(d) > 0.5 {
		t.Errorf("1kHz in an uncompressed band changed by %+.2fdB", d)
	}
}

func maxAbs(x []float64) float64 {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, math.Abs(v))
	}
	return m
}

func TestTuningPicksTheNearestFIR(t *testing.T) {
	tn := newTuning(&TuningSpec{FIRs: []tuningFIR{{30, []float64{1}}, {60, []float64{1}}, {100, []float64{1}}}}, 48000)
	for pct, want := range map[int]int{0: 0, 30: 0, 44: 0, 46: 1, 60: 1, 79: 1, 81: 2, 100: 2} {
		if got := tn.firFor(pct); got != want {
			t.Errorf("volume %d%%: FIR %d, want %d", pct, got, want)
		}
	}
}

func TestLoadTuningReadsWhatIsThere(t *testing.T) {
	dir := t.TempDir()
	write := func(name, data string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if s, err := LoadTuning(dir); err != nil || !s.Empty() {
		t.Fatalf("an empty dir must give an empty spec, got %v %v", s, err)
	}
	write("ParametricEQ.cfg", testPEQ)
	write("EQ_60.cfg", "1.0,\n0.0,\n")
	write("EQ_30.cfg", "0.5,\n")
	write("EQ_notes.cfg", "ignored")
	write("MBCL.cfg", testMBCL)
	s, err := LoadTuning(dir)
	if err != nil || len(s.PEQ) != 2 || len(s.FIRs) != 2 || s.FIRs[0].Percent != 30 || s.MBCL == nil {
		t.Fatalf("spec=%+v err=%v", s, err)
	}
	write("MBCL.cfg", "{ broken")
	if _, err := LoadTuning(dir); err == nil {
		t.Fatal("a present but unreadable file must be an error, not a partial tuning")
	}
}

// monoPeriod is a dual-mono S16 period of a sine.
func monoPeriod(f, amp float64, frames, start int) []byte {
	return interleave(frames, start, tone(amp, f), tone(amp, f))
}

func periodRMS(b []byte) float64 {
	l := channel(b, 0)
	s := 0.0
	for _, v := range l {
		s += float64(v) * float64(v)
	}
	return math.Sqrt(s / float64(len(l)))
}

func tunedChain(spec *TuningSpec, p Params) *Chain {
	c := New(48000)
	c.SetActive(true)
	c.SetParams(p)
	if spec != nil {
		c.SetTuning(spec)
	}
	return c
}

// The stage runs on mono speaker audio, and not when Params.SpeakerTuning is
// off (the jack) or the audio is stereo.
func TestChainRunsTheTuningOnlyOnTheMonoSpeakerPath(t *testing.T) {
	spec := &TuningSpec{PEQ: []tuningBiquad{{"PEAK", 1000, 1, 6}}}
	flat := DefaultParams()
	flat.LimiterEnabled = false

	level := func(c *Chain, stereo bool) float64 {
		var last []byte
		for p := 0; p < 6; p++ {
			var buf []byte
			if stereo {
				buf = interleave(2048, p*2048, tone(3000, 1000), tone(3000*0.9, 1000))
			} else {
				buf = monoPeriod(1000, 3000, 2048, p*2048)
			}
			c.Process(buf)
			last = buf
		}
		return periodRMS(last)
	}

	plain := level(tunedChain(nil, flat), false)
	on := level(tunedChain(spec, flat), false)
	if d := 20 * math.Log10(on/plain); math.Abs(d-6) > 0.3 {
		t.Errorf("tuning on: %+.2fdB at 1kHz, want +6", d)
	}
	off := flat
	off.SpeakerTuning = false
	if d := 20 * math.Log10(level(tunedChain(spec, off), false)/plain); math.Abs(d) > 0.01 {
		t.Errorf("SpeakerTuning off: %+.2fdB, want 0", d)
	}
	plainStereo := level(tunedChain(nil, flat), true)
	if d := 20 * math.Log10(level(tunedChain(spec, flat), true)/plainStereo); math.Abs(d) > 0.01 {
		t.Errorf("stereo: %+.2fdB, want 0 — the tuning is for the mono speaker", d)
	}
}

// While the tuning runs it is the driver's protection, so the bass guard
// stays out of the way.
func TestChainSkipsTheGuardWhileTheTuningRuns(t *testing.T) {
	p := DefaultParams() // guard on
	p.LimiterEnabled = false
	run := func(c *Chain) float64 {
		for k := 0; k < 8; k++ {
			c.Process(monoPeriod(50, 20000, 2048, k*2048))
		}
		return c.TakeStats().GuardReductionDb
	}
	if r := run(tunedChain(nil, p)); r <= 0 {
		t.Fatalf("control: the guard should reduce loud 50Hz, got %.2fdB", r)
	}
	spec := &TuningSpec{PEQ: []tuningBiquad{{"PEAK", 1000, 1, 0.1}}}
	if r := run(tunedChain(spec, p)); r != 0 {
		t.Errorf("guard reduced %.2fdB while the tuning ran, want 0", r)
	}
}

// The compressor works at the playback level: loud bass that it pulls down at
// full volume passes untouched at a low one, back at full scale for the
// volume stage after the chain.
func TestTuningCompressorSeesThePlaybackLevel(t *testing.T) {
	m, _ := parseMBCL([]byte(testMBCL))
	spec := &TuningSpec{MBCL: m}
	in := sine(40, fullScale*0.5, 48000) // -6 dBFS
	run := func(gain float64) float64 {
		tn := newTuning(spec, 48000)
		x := append([]float64(nil), in...)
		tn.process(x, 50, gain)
		return 20 * math.Log10(rms(x[24000:])/rms(in[24000:]))
	}
	if d := run(1); d > -3 {
		t.Errorf("at full volume 40Hz at -6dBFS changed by %+.1fdB, want it compressed", d)
	}
	if d := run(dbToGain(-30)); math.Abs(d) > 0.5 {
		t.Errorf("at -30dB volume it changed by %+.1fdB, want ~0: the driver gets -36dBFS", d)
	}
}
