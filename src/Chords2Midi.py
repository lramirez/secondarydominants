import numpy as np
from midiutil import MIDIFile
from sdpychord import pychord as pc
NOTES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DATA_PATH = "../data/chords3.csv"
MODEL_PATH = "../models/model.pt"
CHAIN_ORDER = 3
N_CANDIDATES = 10000
tempo = 80
track = 0
channel = 0
volume   = 100
chord_duration = 4
bass_duration = 1
arp_duration = 0.5
eight_times = [0, 0.15]

# This function was borrowed from stackexchange. Kudos for the author.
def NoteToMidi(KeyOctave):
    # KeyOctave is formatted like 'C#3'
    key = KeyOctave[:-1]  # eg C, Db
    octave = KeyOctave[-1]   # eg 3, 4
    answer = -1

    try:
        if 'b' in key:
            pos = NOTES_FLAT.index(key)
        else:
            pos = NOTES_SHARP.index(key)
    except:
        print('The key is not valid', key)
        return answer

    answer += pos + 12 * (int(octave) + 1) + 1
    return answer

# This is clunky as hell. Needs serious improvement.

def render(chords, filename):
    chords = [pc.Chord(c) for c in chords]
    MIDI = MIDIFile(1)
    for bar in range(len(chords)):
        chordtones = chords[bar].components_with_pitch(root_pitch=3)
        basstones = chords[bar].components_with_pitch(root_pitch=1)[:3]
        bt = chords[bar].components_with_pitch(root_pitch=1)[:2]
        bt.reverse()
        basstones = basstones+bt
        arp = Arpeggiate(chords[bar])
        for ct in chordtones:
            MIDI.addNote(track, channel, NoteToMidi(ct), bar*4, chord_duration, volume)
        if len(basstones) < 4:
            basstones = basstones + [basstones[1]]
        for i in range(4):
            MIDI.addNote(track, channel, NoteToMidi(basstones[i]), bar*4 + i*bass_duration, bass_duration, volume)
        for i in range(8):
            MIDI.addNote(track, channel, NoteToMidi(arp.__next__()), bar*4+i*0.5+eight_times[i%2], arp_duration, volume)
    with open(filename, "wb") as output_file:
        MIDI.writeFile(output_file)    

class Arpeggiate:
    def __init__(self, chord, reverse=False):
        ct = chord.components_with_pitch(root_pitch=4)
        if reverse:
            ct.reverse()
        chordtones = [] + ct
        rct = ct[:-1]
        rct.reverse()
        chordtones = chordtones+rct
        self.ct = chordtones
        self.current = 0

    def __iter__(self):
        return self
    def __next__(self):
        self.current += 1
        return self.ct[(self.current-1) % len(self.ct)]
    