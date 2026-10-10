package outchain

import "math"

// fftPlan is an iterative radix-2 complex FFT of one fixed power-of-two size,
// for the tuning stage's FIR convolution. Twiddles and the bit-reversal
// permutation are computed once; transform allocates nothing.
type fftPlan struct {
	n        int
	cos, sin []float64 // twiddles for the forward transform, n/2 of each
	rev      []int
}

func newFFTPlan(n int) *fftPlan {
	if n < 2 || n&(n-1) != 0 {
		panic("outchain: FFT size must be a power of two")
	}
	p := &fftPlan{n: n, cos: make([]float64, n/2), sin: make([]float64, n/2), rev: make([]int, n)}
	for i := 0; i < n/2; i++ {
		a := -2 * math.Pi * float64(i) / float64(n)
		p.cos[i], p.sin[i] = math.Cos(a), math.Sin(a)
	}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for i := 0; i < n; i++ {
		r := 0
		for b := 0; b < bits; b++ {
			if i&(1<<b) != 0 {
				r |= 1 << (bits - 1 - b)
			}
		}
		p.rev[i] = r
	}
	return p
}

// transform runs the FFT in place on (re, im). The inverse is unscaled; the
// caller divides by n.
func (p *fftPlan) transform(re, im []float64, inverse bool) {
	n := p.n
	for i, r := range p.rev {
		if i < r {
			re[i], re[r] = re[r], re[i]
			im[i], im[r] = im[r], im[i]
		}
	}
	sign := 1.0
	if inverse {
		sign = -1
	}
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		step := n / size
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				wr, wi := p.cos[k*step], sign*p.sin[k*step]
				a, b := start+k, start+k+half
				tr := wr*re[b] - wi*im[b]
				ti := wr*im[b] + wi*re[b]
				re[b], im[b] = re[a]-tr, im[a]-ti
				re[a], im[a] = re[a]+tr, im[a]+ti
			}
		}
	}
}
