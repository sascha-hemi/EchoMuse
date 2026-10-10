//go:build server

package speaker

import (
	"log"
	"os"
	"time"

	"github.com/wilbowes/EchoMuse/internal/bindings/mixer"
	"github.com/wilbowes/EchoMuse/internal/outchain"
)

// Radar adds a physical mute and a longer settling sequence. Keep those
// requirements separate from the shared PCM loop and other boards' timing.
func (p *PcmSpeaker) startRadarOutput() error {
	readStatus := func() ([]byte, error) { return os.ReadFile(p.statusFile) }
	if err := waitForRunningPCM(readStatus, p.deadCh, 3*time.Second); err != nil {
		return err
	}
	return unmuteRadarSpeaker(func(d time.Duration) error { return waitForSilence(p.deadCh, d) })
}

// Do not close the native PCM while its writer is still using it.
func (p *PcmSpeaker) abortRadarStartup() {
	mixer.Set(radarMute, "On")
	mixer.Set(mixer.PlaybackVolume, "0")
	mixer.Set(mixer.SpeakerAmp, "Off")
	if p.session != nil {
		close(p.stopCh)
		<-p.deadCh
		p.session.Close()
	}
}

// radarTuningDir holds the stock audio service's speaker tuning, and
// radarTuningOff, when it exists, keeps the tuning stage off for an A/B
// comparison (touch it and restart the server).
const (
	radarTuningDir = "/system/vendor/etc/audio-algorithms"
	radarTuningOff = "/data/emos/no-amazon-tuning"
)

// loadRadarTuning installs the stock speaker tuning in the output chain
// (outchain/tuning.go): read from the device, never bundled. Any failure
// leaves the chain as it was, which is the sound every build before this one
// had.
func (p *PcmSpeaker) loadRadarTuning() {
	if _, err := os.Stat(radarTuningOff); err == nil {
		log.Printf("[speaker] Radar speaker tuning off (%s exists)", radarTuningOff)
		return
	}
	spec, err := outchain.LoadTuning(radarTuningDir)
	if err != nil {
		log.Printf("[speaker] Radar speaker tuning unavailable: %v", err)
		return
	}
	if spec.Empty() {
		log.Printf("[speaker] Radar speaker tuning: nothing in %s", radarTuningDir)
		return
	}
	p.chain.SetTuning(spec)
	log.Printf("[speaker] Radar speaker tuning loaded from %s: %s (bass guard skipped while it runs)", radarTuningDir, spec)
}
