package outchain

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// shaped is a chain doing real work on every stage: boosted EQ (so the
// sections carry state), the bass guard engaged by a loud low band, and a
// limiter low enough to reduce.
func shaped() *Chain {
	c := New(48000)
	c.SetActive(true)
	p := DefaultParams()
	p.Bands = [NumBands]float64{9, 6, 0, -3, 0, 3, 6, 9}
	p.GuardEnabled, p.GuardDb = true, -12
	p.LimiterEnabled, p.LimiterThresholdDb = true, -6
	c.SetParams(p)
	return c
}

// interleave builds a stereo S16_LE period from two channel generators.
func interleave(frames, start int, l, r func(i int) float64) []byte {
	b := make([]byte, frames*4)
	for i := 0; i < frames; i++ {
		binary.LittleEndian.PutUint16(b[i*4:], uint16(int16(l(start+i))))
		binary.LittleEndian.PutUint16(b[i*4+2:], uint16(int16(r(start+i))))
	}
	return b
}

func tone(amp, hz float64) func(int) float64 {
	return func(i int) float64 { return amp * math.Sin(2*math.Pi*hz*float64(i)/48000) }
}

func silence(int) float64 { return 0 }

func channel(b []byte, ch int) []int16 {
	out := make([]int16, len(b)/4)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*4+ch*2:]))
	}
	return out
}

// The switch to stereo processing must be inaudible: on dual-mono audio the
// stereo path computes exactly what the mono path does, so a chain that has
// been stereo from the first sample and one that switches on the first
// differing period agree bit for bit, before and after the switch.
func TestSwitchToStereoIsBitExact(t *testing.T) {
	mono, forced := shaped(), shaped()
	// One silent period each, so activation's reset is behind them, then the
	// second goes stereo from zero state — what a split would copy.
	mono.Process(make([]byte, 2048*4))
	forced.Process(make([]byte, 2048*4))
	forced.stereo = true

	bass := tone(20000, 60)
	for p := 0; p < 6; p++ {
		a := interleave(2048, p*2048, bass, bass)
		b := append([]byte(nil), a...)
		mono.Process(a)
		forced.Process(b)
		if mono.stereo {
			t.Fatalf("period %d: dual-mono audio switched the chain to stereo", p)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("period %d: the stereo path differs from the mono path on L == R", p)
		}
	}

	// Now the switch itself, mid-stream with every filter carrying state.
	a := interleave(2048, 6*2048, bass, tone(9000, 1000))
	b := append([]byte(nil), a...)
	mono.Process(a)
	forced.Process(b)
	if !mono.stereo {
		t.Fatal("did not switch on the first period with L != R")
	}
	if !bytes.Equal(a, b) {
		t.Fatal("switching mid-stream is not the same as having been stereo all along")
	}
}

// What is in one channel stays in that channel.
func TestStereoKeepsTheChannelsApart(t *testing.T) {
	c := New(48000)
	c.SetActive(true) // defaults: flat EQ, guard and limiter idle at this level
	for p := 0; p < 4; p++ {
		buf := interleave(2048, p*2048, tone(8000, 440), silence)
		c.Process(buf)
		for i, s := range channel(buf, 1) {
			if s != 0 {
				t.Fatalf("period %d frame %d: the right channel carries %d of the left", p, i, s)
			}
		}
	}
}

// One gain for both sides: a peak on the left pulls the right down by the
// same amount, so a limited passage does not shift towards the quiet side.
func TestLimiterIsLinked(t *testing.T) {
	c := New(48000)
	c.SetActive(true)
	p := DefaultParams()
	p.LimiterEnabled, p.LimiterThresholdDb = true, -12
	c.SetParams(p)
	var l, r []int16
	for k := 0; k < 4; k++ {
		buf := interleave(2048, k*2048, func(int) float64 { return 30000 }, func(int) float64 { return 3000 })
		c.Process(buf)
		l, r = channel(buf, 0), channel(buf, 1)
	}
	last := len(l) - 1
	if l[last] >= 30000 {
		t.Fatalf("the left was not limited: %d", l[last])
	}
	ratioL := float64(l[last]) / 30000
	ratioR := float64(r[last]) / 3000
	if math.Abs(ratioL-ratioR) > 0.002 {
		t.Errorf("unlinked: left at %.4f of input, right at %.4f", ratioL, ratioR)
	}
}

// Silence resets the chain, and with it the stereo state: what follows is
// processed as mono again until it shows otherwise.
func TestStereoEndsWithTheResetOnSilence(t *testing.T) {
	c := shaped()
	c.Process(interleave(2048, 0, tone(9000, 440), silence))
	if !c.stereo {
		t.Fatal("not stereo")
	}
	for n := 0; !c.Idle(); n++ {
		if n > 200 {
			t.Fatal("never went idle on silence")
		}
		c.Process(make([]byte, 2048*4))
	}
	if c.stereo {
		t.Error("still stereo after the reset")
	}
}

// wideOf is a period as the boosted-response path hands it over: the same
// samples scaled up by scale, with one full-scale multiplier per frame.
func wideOf(b []byte, scale float64) (wide, scales []float64) {
	frames := len(b) / 4
	wide, scales = make([]float64, frames*2), make([]float64, frames)
	for i := 0; i < frames; i++ {
		wide[i*2] = float64(int16(binary.LittleEndian.Uint16(b[i*4:]))) * scale
		wide[i*2+1] = float64(int16(binary.LittleEndian.Uint16(b[i*4+2:]))) * scale
		scales[i] = scale
	}
	return wide, scales
}

// A boosted reply is mixed over the music on the wide path. Over stereo
// music that period must stay stereo too, or the image collapses for as long
// as the reply plays.
func TestWidePeriodKeepsTheChannelsApart(t *testing.T) {
	c := New(48000)
	c.SetActive(true)
	wide, scales := wideOf(interleave(2048, 0, tone(8000, 440), silence), 4)
	c.ProcessFloat(wide, scales)
	if !c.stereo {
		t.Fatal("a wide period with L != R did not switch the chain to stereo")
	}
	for i := 0; i < len(scales); i++ {
		if wide[i*2+1] != 0 {
			t.Fatalf("frame %d: the right channel carries %.3f of the left", i, wide[i*2+1])
		}
	}
}

// Dual-mono on the wide path stays on the mono path — a reply over nothing,
// or over mono music, is processed exactly as before.
func TestWideDualMonoStaysMono(t *testing.T) {
	c := shaped()
	bass := tone(12000, 60)
	wide, scales := wideOf(interleave(2048, 0, bass, bass), 4)
	c.ProcessFloat(wide, scales)
	if c.stereo {
		t.Fatal("dual-mono switched the chain to stereo")
	}
}

// Ordinary and wide periods share one stereo state: a wide period in the
// middle of stereo music computes what Process would have (to within the
// truncation the wide path skips), and the ordinary period after it is
// bit-exact to a chain that never left Process — so the right channel's
// filters were carried through, not frozen or overwritten by the left's.
func TestWidePeriodInStereoMatchesProcess(t *testing.T) {
	reference, switched := shaped(), shaped()
	l, r := tone(14000, 60), tone(9000, 1000)
	for chunk := 0; chunk < 4; chunk++ {
		want := interleave(2048, chunk*2048, l, r)
		in := append([]byte(nil), want...)
		reference.Process(want)

		if chunk == 2 {
			const scale = 4.0
			wide, scales := wideOf(in, scale)
			switched.ProcessFloat(wide, scales)
			for i := range scales {
				for ch := 0; ch < 2; ch++ {
					got := wide[i*2+ch] / scale
					exp := float64(int16(binary.LittleEndian.Uint16(want[i*4+ch*2:])))
					if math.Abs(got-exp) > 1 {
						t.Fatalf("wide frame %d ch %d: normalised %.3f, want %.3f", i, ch, got, exp)
					}
				}
			}
			continue
		}

		switched.Process(in)
		if !bytes.Equal(in, want) {
			t.Fatalf("ordinary chunk %d differs from a chain that stayed on Process", chunk)
		}
	}
	if !switched.stereo {
		t.Fatal("not stereo")
	}
}
