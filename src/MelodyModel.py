import pandas as pd
import numpy as np
import markovify
import ast
import random
import operator
import bisect
import json
import copy

BEGIN = "___BEGIN__"
END = "___END__"

class MelodyChain(markovify.Chain):
    def __init__(self, data, state_size, all_chords, notes_for_chord):
        self.notes_for_chord = notes_for_chord
        self.all_chords = all_chords

        super().__init__(data, state_size=state_size)


    def gen(self, chords, init_state=None):
        state = init_state or (markovify.chain.BEGIN,) * self.state_size
        n = len(chords)
        i = 0

        while i < n:
            next_word = self.move(state, chords[i])
            yield next_word
            state = tuple(state[1:]) + (next_word,)
            i += 1


    def move(self, state, chord):
        """
        Given a state, choose the next item at random.
        """
        if self.compiled:
            choices, cumdist = self.model[state]
        elif state == tuple([ markovify.chain.BEGIN ] * self.state_size):
            choices = self.begin_choices
            cumdist = self.begin_cumdist
        else:
            choices, weights = zip(*self.model[state].items())
            cumdist = list(markovify.chain.accumulate(weights))
        
        choices = list(choices)
        
        rem = []
        if chord in self.all_chords:
            i = 0
            for c in choices:
                if not c in self.notes_for_chord[chord]:
                    rem.append(i)
                i += 1

        else:
            if END in choices:
                rem.append(choices.index(END))

        for i in sorted(rem, reverse=True):
            if len(cumdist) > 1:
                del choices[i]
                if i < len(cumdist):
                    prev = 0
                    if i > 0:
                        prev = cumdist[i - 1]

                    val = cumdist[i] - prev
                    
                    for j in range(i + 1, len(cumdist)):
                        cumdist[j] -= val
                    
                    del cumdist[i]            

        r = random.random() * cumdist[-1]
        selection = choices[bisect.bisect(cumdist, r)]
        return selection

    def from_json(cls, json_thing, all_chords, notes_for_chord):
        """
        Given a JSON object or JSON string that was created by `self.to_json`,
        return the corresponding markovify.Chain.
        """

        if isinstance(json_thing, basestring):
            obj = json.loads(json_thing)
        else:
            obj = json_thing

        if isinstance(obj, list):
            rehydrated = dict((tuple(item[0]), item[1]) for item in obj)
        elif isinstance(obj, dict):
            rehydrated = obj
        else:
            raise ValueError("Object should be dict or list")

        state_size = len(list(rehydrated.keys())[0])

        inst = cls(None, state_size, rehydrated)

        inst.notes_for_chord = notes_for_chord
        inst.all_chords = all_chords

        return inst



class MelodyModel:
    def __init__(self, data_path="../data/melody_chords.csv", model_path="../models/melodies_markov.json"):
        df = pd.read_csv(data_path)

        all_chords = self.__get_all_chords(df)
        notes_for_chord = self.__get_notes_for_chord(df, all_chords)

        data = self.__get_data(df)

        for track_id in df["track_id"].unique():
            track_df = df[df["track_id"] == track_id]

            notes = track_df["pitch"].tolist()

            data.append(notes)
        
        # TODO: fix model importing so retraining is not required every time
        self.model = MelodyChain(data, state_size=2, all_chords=all_chords, notes_for_chord=notes_for_chord)


    def get_melody_for_chords(self, chords):
        prog = []
        for c in chords:
            prog.append(c[1])

        notes = [x for x in self.model.gen(prog)]

        return notes


    def __get_all_chords(self, df):
        all_chords = set(df["chord"].unique())
        return all_chords
        
        
    def __get_notes_for_chord(self, df, all_chords):
        notes_for_chord = {}

        for chord in all_chords:
            chord_df = df[df["chord"] == chord]
            notes = set(chord_df["pitch"].unique())

            notes_for_chord[chord] = notes
        
        return notes_for_chord


    def __get_data(self, df):
        data = []

        for track_id in df["track_id"].unique():
            track_df = df[df["track_id"] == track_id]

            notes = track_df["pitch"].tolist()

            data.append(notes)

        return data