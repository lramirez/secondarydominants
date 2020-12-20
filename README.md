# secondarydominants
Disclaimer: this readme is still WIP.

This is a course project for Computational Creativity 2020.

The system generates jazz tunes by first generating chord sequences, and then using the chord tones, bass and melody arpeggios on top.
The chord sequences are generated using markov chains, the outputs of which are scored using a LSTM language model. Both are trained using (insert dataset name here).

Our system uses a modified version of [pychord][https://github.com/yuma-m/pychord]. The library doesn't support certain chord qualities yet, so we had to add them ourselves.

Installation:
1. Clone this repository
2. Clone pychord
3. Replace the file pychord/constants/qualities.py with our version of it.
4. Run an enjoy. (TODO: finish this) We intend to implement audio artifact rendering to replace simple midi files. Meanwhile, we recommend using a DAW to play the midi files with instruments of choice. Or, give the text output of the chord sequences to autobop (TODO: add link), and watch the magic happen.
  
