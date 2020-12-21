# secondarydominants

This is a course project for Computational Creativity 2020.

The system generates AABA-structured jazz tunes by first generating chord sequences and simple melody tones. It then generates midi files, which contain chords, bass and arpeggios from the chord tones, and a simple brass section using the melody tones.
The chord sequences are generated using markov chains, the outputs of which are scored using a LSTM language model. Both the chord MC and the LSTM model are trained using the [iReal Jazz corpus](https://www.musiccognition.osu.edu/resources/). The melody tones are also generated by a Markov chain, trained using the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/dbformat/dboverview.html).

Our system uses a modified version of [pychord](https://github.com/yuma-m/pychord). The library doesn't support certain chord qualities yet, so we had to add them ourselves.

Installation:
1. Clone this repository
2. Clone pychord
3. Replace the file pychord/constants/qualities.py with our version of it.
5. Copy the modified pychord into src/sdpychord.
5. Run and enjoy. Usage: python JazzGenerator.py -o <desired filename for the midi file>

We intend to implement audio artifact rendering to replace simple midi files. Meanwhile, we recommend using a DAW to play the midi files with instruments of choice. Or, give the text output of the chord sequences to [Autobop](https://github.com/HajimeKawahara/autobop) and watch the magic happen.
  
The demos folder contains a few selected outputs of the system rendered to audio.
