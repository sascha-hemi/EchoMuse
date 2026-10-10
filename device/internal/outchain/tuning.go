package outchain

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// The speaker tuning stage reproduces the processing the stock audio service
// (/system/bin/mixer) applies to the internal speaker, from that service's own
// configuration on the device's system partition:
//
//   - ParametricEQ.cfg, a fixed set of biquads;
//   - EQ_<volume>.cfg, FIR filters chosen by volume percent;
//   - MBCL.cfg, a four-band compressor-limiter.
//
// Nothing of it is in this repository. The files are read at runtime from the
// device that is running, as the codec filter profiles are (bindings/speaker
// radar.go), so a build distributes no vendor tuning. What the files do not
// say is an assumption, named where it is made: the order of the three
// stages, the volume each EQ_<n> applies at, and the compressor time
// constants. On Radar1 (2026-10-10) the speaker had no audible bass without
// this processing, though the woofer played a 60Hz tone driven directly.

// TuningSpec is the parsed configuration. Any part may be absent.
type TuningSpec struct {
	PEQ  []tuningBiquad
	FIRs []tuningFIR // sorted by Percent
	MBCL *mbclSpec
}

type tuningBiquad struct {
	FilterType string  `json:"FilterType"`
	Fc         float64 `json:"Fc"`
	Q          float64 `json:"Q"`
	GaindB     float64 `json:"GaindB"`
}

type tuningFIR struct {
	Percent int
	Taps    []float64
}

// Empty reports whether the spec would change nothing.
func (s *TuningSpec) Empty() bool {
	return s == nil || (len(s.PEQ) == 0 && len(s.FIRs) == 0 && s.MBCL == nil)
}

// String summarises what was loaded, for the log.
func (s *TuningSpec) String() string {
	if s.Empty() {
		return "none"
	}
	var pct []string
	for _, f := range s.FIRs {
		pct = append(pct, strconv.Itoa(f.Percent))
	}
	return fmt.Sprintf("peq=%d biquads, fir=[%s]%%, mbcl=%v",
		len(s.PEQ), strings.Join(pct, ","), s.MBCL != nil)
}

// LoadTuning reads the tuning files from dir. A missing file leaves its part
// out; a file that is present but unreadable is an error, because a partial
// tuning is a different sound from the one the files describe.
func LoadTuning(dir string) (*TuningSpec, error) {
	s := &TuningSpec{}
	var errs []error

	if data, err := os.ReadFile(filepath.Join(dir, "ParametricEQ.cfg")); err == nil {
		peq, err := parsePEQ(data)
		if err != nil {
			errs = append(errs, fmt.Errorf("ParametricEQ.cfg: %w", err))
		}
		s.PEQ = peq
	}

	names, _ := filepath.Glob(filepath.Join(dir, "EQ_*.cfg"))
	for _, name := range names {
		pct, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(filepath.Base(name), "EQ_"), ".cfg"))
		if err != nil {
			continue // not a volume table
		}
		data, err := os.ReadFile(name)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		taps, err := parseFIR(data)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", filepath.Base(name), err))
			continue
		}
		s.FIRs = append(s.FIRs, tuningFIR{Percent: pct, Taps: taps})
	}
	sort.Slice(s.FIRs, func(i, j int) bool { return s.FIRs[i].Percent < s.FIRs[j].Percent })

	if data, err := os.ReadFile(filepath.Join(dir, "MBCL.cfg")); err == nil {
		m, err := parseMBCL(data)
		if err != nil {
			errs = append(errs, fmt.Errorf("MBCL.cfg: %w", err))
		}
		s.MBCL = m
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}
	return s, nil
}

// stripComments removes /* */ and // comments outside strings, which these
// files carry around otherwise-valid JSON.
func stripComments(data []byte) []byte {
	var out []byte
	inStr, esc := false, false
	for i := 0; i < len(data); i++ {
		c := data[i]
		if inStr {
			out = append(out, c)
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			out = append(out, c)
			continue
		}
		if c == '/' && i+1 < len(data) && data[i+1] == '/' {
			for i < len(data) && data[i] != '\n' {
				i++
			}
			out = append(out, '\n')
			continue
		}
		if c == '/' && i+1 < len(data) && data[i+1] == '*' {
			i += 2
			for i+1 < len(data) && !(data[i] == '*' && data[i+1] == '/') {
				i++
			}
			i++ // past '/'
			continue
		}
		out = append(out, c)
	}
	return out
}

func parsePEQ(data []byte) ([]tuningBiquad, error) {
	var f struct {
		Bypass  bool           `json:"Bypass"`
		Biquads []tuningBiquad `json:"Biquad Definitions"`
	}
	if err := json.Unmarshal(stripComments(data), &f); err != nil {
		return nil, err
	}
	if f.Bypass {
		return nil, nil
	}
	var out []tuningBiquad
	for _, b := range f.Biquads {
		switch b.FilterType {
		case "BYPASS":
			continue
		case "PEAK", "LOW_SHELF", "HIGH_SHELF", "LOW_PASS", "HIGH_PASS":
		default:
			return nil, fmt.Errorf("unsupported filter type %q", b.FilterType)
		}
		if b.Fc <= 0 || b.Q <= 0 {
			return nil, fmt.Errorf("%s at %gHz: invalid Fc/Q", b.FilterType, b.Fc)
		}
		out = append(out, b)
	}
	return out, nil
}

var firNumber = regexp.MustCompile(`[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?`)

func parseFIR(data []byte) ([]float64, error) {
	var taps []float64
	for _, s := range firNumber.FindAll(stripComments(data), -1) {
		v, err := strconv.ParseFloat(string(s), 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("invalid tap %q", s)
		}
		taps = append(taps, v)
	}
	if len(taps) == 0 {
		return nil, errors.New("no taps")
	}
	if len(taps) > firMaxTaps {
		return nil, fmt.Errorf("%d taps, at most %d supported", len(taps), firMaxTaps)
	}
	return taps, nil
}

func parseMBCL(data []byte) (*mbclSpec, error) {
	var m mbclSpec
	if err := json.Unmarshal(stripComments(data), &m); err != nil {
		return nil, err
	}
	if m.Bypass {
		return nil, nil
	}
	if len(m.FC) != 3 || len(m.Bands) != 4 {
		return nil, fmt.Errorf("want 3 crossover frequencies and 4 bands, got %d and %d", len(m.FC), len(m.Bands))
	}
	for i, fc := range m.FC {
		if fc <= 0 || (i > 0 && fc <= m.FC[i-1]) {
			return nil, fmt.Errorf("crossover frequencies must rise: %v", m.FC)
		}
	}
	return &m, nil
}

// designTuningBiquad is the Audio EQ Cookbook with the file's Q, including
// the shelves, whose Q the graphic EQ's designs fix at S=1.
func designTuningBiquad(b tuningBiquad, fs float64) biquad {
	A := math.Pow(10, b.GaindB/40)
	w0 := 2 * math.Pi * b.Fc / fs
	cw, sw := math.Cos(w0), math.Sin(w0)
	alpha := sw / (2 * b.Q)
	switch b.FilterType {
	case "PEAK":
		return peaking(b.Fc, b.GaindB, b.Q, fs)
	case "LOW_SHELF":
		sq := 2 * math.Sqrt(A) * alpha
		return norm(A*((A+1)-(A-1)*cw+sq), 2*A*((A-1)-(A+1)*cw), A*((A+1)-(A-1)*cw-sq),
			(A+1)+(A-1)*cw+sq, -2*((A-1)+(A+1)*cw), (A+1)+(A-1)*cw-sq)
	case "HIGH_SHELF":
		sq := 2 * math.Sqrt(A) * alpha
		return norm(A*((A+1)+(A-1)*cw+sq), -2*A*((A-1)+(A+1)*cw), A*((A+1)+(A-1)*cw-sq),
			(A+1)-(A-1)*cw+sq, 2*((A-1)-(A+1)*cw), (A+1)-(A-1)*cw-sq)
	case "LOW_PASS":
		return norm((1-cw)/2, 1-cw, (1-cw)/2, 1+alpha, -2*cw, 1-alpha)
	case "HIGH_PASS":
		return norm((1+cw)/2, -(1 + cw), (1+cw)/2, 1+alpha, -2*cw, 1-alpha)
	}
	return biquad{b0: 1}
}

// tuning is the running stage: PEQ, then the volume FIR, then the MBCL. The
// order is an assumption (the first two are linear and commute; dynamics go
// last, as they must, to catch what the EQ boosts).
type tuning struct {
	peq     []biquad
	fir     *firConv
	firPct  []int
	mbcl    *mbcl
	scratch []float64
}

func newTuning(s *TuningSpec, fs float64) *tuning {
	t := &tuning{}
	for _, b := range s.PEQ {
		t.peq = append(t.peq, designTuningBiquad(b, fs))
	}
	if len(s.FIRs) > 0 {
		var taps [][]float64
		for _, f := range s.FIRs {
			taps = append(taps, f.Taps)
			t.firPct = append(t.firPct, f.Percent)
		}
		t.fir = newFIRConv(taps)
	}
	if s.MBCL != nil {
		t.mbcl = newMBCL(s.MBCL, fs)
	}
	return t
}

// firFor picks the FIR for a volume percent: the one whose percent is
// nearest, the quietest below the quietest table and the loudest above the
// loudest. That EQ_<n> applies at volume n% is an assumption from the names.
func (t *tuning) firFor(pct int) int {
	best, bestD := 0, math.MaxInt
	for i, p := range t.firPct {
		d := p - pct
		if d < 0 {
			d = -d
		}
		if d < bestD {
			best, bestD = i, d
		}
	}
	return best
}

// process runs the stage over a mono block in place.
//
// gain is the playback volume that will be applied after the chain. The
// compressor-limiter runs on the signal at that level and hands it back at
// full scale: it is protecting the driver from what the driver will actually
// get. Run on the full-scale signal instead, it pulled the boosted bass down
// at every volume, since our volume comes after it (Radar1, 2026-10-10: "the
// deep bass is still missing"). That the stock chain sets its level the same
// way is inferred, not read: MBCL.cfg calls its input level "system gain".
func (t *tuning) process(x []float64, volumePct int, gain float64) {
	if len(t.peq) > 0 {
		for i, v := range x {
			for k := range t.peq {
				v = t.peq[k].step(v)
			}
			x[i] = v
		}
	}
	if t.fir != nil {
		t.fir.process(x, t.firFor(volumePct))
	}
	if t.mbcl != nil {
		if gain < 1e-6 || gain > 1 {
			gain = 1 // muted: nothing will be heard, so do not divide by it
		}
		for i, v := range x {
			x[i] = t.mbcl.step(v*gain) / gain
		}
	}
}

func (t *tuning) reset() {
	for k := range t.peq {
		t.peq[k].reset()
	}
	if t.fir != nil {
		t.fir.reset()
	}
	if t.mbcl != nil {
		t.mbcl.reset()
	}
}

// block returns a scratch buffer of n samples, reused across periods.
func (t *tuning) block(n int) []float64 {
	if cap(t.scratch) < n {
		t.scratch = make([]float64, n)
	}
	return t.scratch[:n]
}
