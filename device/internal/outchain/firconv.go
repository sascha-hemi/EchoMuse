package outchain

// firConvN is the FFT size of the tuning FIR: room for a 2048-tap filter and
// up to 2049 new samples per transform. Overlap-save with a fixed size keeps
// the cost per period constant and adds no latency — each call's output is
// complete when it returns.
const (
	firConvN   = 4096
	firMaxTaps = 2048
)

// firConv runs one of several FIR filters over a mono stream by overlap-save
// FFT convolution. A 2048-tap filter applied directly is ~98 million
// multiply-adds a second at 48kHz; this is two 4096-point FFTs per call.
//
// Switching filter (the tuning FIRs are chosen by volume) crossfades over one
// call: both filters run on the same input history, so the old and the new
// output are both exact and the fade between them has no discontinuity.
type firConv struct {
	plan     *fftPlan
	spectra  [][2][]float64 // per filter: FFT of the zero-padded taps
	hist     []float64      // the last firConvN input samples, oldest first
	re, im   []float64
	re2, im2 []float64
	cur      int // filter in use; -1 before the first call
}

func newFIRConv(filters [][]float64) *firConv {
	f := &firConv{
		plan: newFFTPlan(firConvN),
		hist: make([]float64, firConvN),
		re:   make([]float64, firConvN), im: make([]float64, firConvN),
		re2: make([]float64, firConvN), im2: make([]float64, firConvN),
		cur: -1,
	}
	for _, taps := range filters {
		re, im := make([]float64, firConvN), make([]float64, firConvN)
		copy(re, taps)
		f.plan.transform(re, im, false)
		f.spectra = append(f.spectra, [2][]float64{re, im})
	}
	return f
}

// process filters x in place with filter want (an index into the filters the
// convolver was built with).
func (f *firConv) process(x []float64, want int) {
	const chunk = firConvN - firMaxTaps + 1
	for len(x) > 0 {
		m := len(x)
		if m > chunk {
			m = chunk
		}
		f.block(x[:m], want)
		x = x[m:]
	}
}

func (f *firConv) block(x []float64, want int) {
	m := len(x)
	copy(f.hist, f.hist[m:])
	copy(f.hist[firConvN-m:], x)

	copy(f.re, f.hist)
	for i := range f.im {
		f.im[i] = 0
	}
	f.plan.transform(f.re, f.im, false)

	if f.cur < 0 {
		f.cur = want
	}
	if want == f.cur {
		f.apply(f.re, f.im, f.cur, x)
		return
	}
	// Crossfade from the old filter to the new over this block.
	copy(f.re2, f.re)
	copy(f.im2, f.im)
	old := make([]float64, m)
	f.apply(f.re2, f.im2, f.cur, old)
	f.apply(f.re, f.im, want, x)
	for i := 0; i < m; i++ {
		t := float64(i+1) / float64(m)
		x[i] = old[i]*(1-t) + x[i]*t
	}
	f.cur = want
}

// apply multiplies the input spectrum (re, im — overwritten) by filter k,
// transforms back, and writes the last len(out) samples, the valid part of
// the circular convolution.
func (f *firConv) apply(re, im []float64, k int, out []float64) {
	hr, hi := f.spectra[k][0], f.spectra[k][1]
	for i := range re {
		r := re[i]*hr[i] - im[i]*hi[i]
		im[i] = re[i]*hi[i] + im[i]*hr[i]
		re[i] = r
	}
	f.plan.transform(re, im, true)
	scale := 1.0 / firConvN
	off := firConvN - len(out)
	for i := range out {
		out[i] = re[off+i] * scale
	}
}

func (f *firConv) reset() {
	for i := range f.hist {
		f.hist[i] = 0
	}
	f.cur = -1
}
