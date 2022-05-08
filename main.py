from mido import MidiFile, Message
import numpy as np
import pretty_midi
from typing import Union, List, Tuple

# divide song on pieces of 1/4 and create sequences of random chords
# each sequence is a list of chords, each chord continues for 1/4
# each sequence is an individual
# each sequence may have chords distributed along whole octaves
# mutation changes several random chords in an individual

minor_keys = {  # but without "m" after main note name
    "C": ["Cm", "Do", "Eb", "Fm", "Gm", "Ab", "Bb"],
    "C#": ["C#m", "D#o", "E", "F#m", "G#m", "A", "B"],
    "D": ["Dm", "Eo", "F", "Gm", "Am", "Bb", "C"],
    "D#": ["D#m", "E#o", "F#", "G#m", "A#m", "B", "C#"],
    "Eb": ["Ebm", "Fo", "Gb", "Abm", "Bbm", "Cb", "Db"],
    "E": ["Em", "F#o", "G", "Am", "Bm", "C", "D"],
    "F": ["Fm", "Go", "Ab", "Bbm", "Cm", "Db", "Eb"],
    "F#": ["F#m", "G#o", "A", "Bm", "C#m", "D", "E"],
    "G": ["Gm", "Ao", "Bb", "Cm", "Dm", "Eb", "F"],
    "G#": ["G#m", "A#o", "B", "C#m", "D#m", "E", "F#"],
    "Ab": ["Abm", "Bbo", "Cb", "Dbm", "Ebm", "Fb", "Gb"],
    "A": ["Am", "Bo", "C", "Dm", "Em", "F", "G"],
    "A#": ["A#m", "B#o", "C#", "D#m", "E#m", "F#", "G#"],
    "Bb": ["Bbm", "Co", "Db", "Ebm", "Fm", "Gb", "Ab"],
    "B": ["Bm", "C#o", "D", "Em", "F#m", "G", "A"]
}

major_keys = {
    "C": ["C", "Dm", "Em", "F", "G", "Am", "Bo"],
    "C#": ["C#", "D#m", "E#m", "F#", "G#", "A#m", "B#o"],
    "Db": ["Db", "Ebm", "Fm", "Gb", "Ab", "Bbm", "Co"],
    "D": ["D", "Em", "F#m", "G", "A", "Bm", "C#o"],
    "D#": ["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Do"],
    "E": ["E", "F#m", "G#m", "A", "B", "C#m", "D#o"],
    "F": ["F", "Gm", "Am", "Bb", "C", "Dm", "Eo"],
    "F#": ["F#", "G#m", "A#m", "B", "C#", "D#m", "E#o"],
    "Gb": ["Gb", "Abm", "Bbm", "Cb", "Db", "Ebm", "Fo"],
    "G": ["G", "Am", "Bm", "C", "D", "Em", "F#o"],
    "G#": ["Ab", "Bbm", "Cm", "Db", "Eb", "Fm", "Go"],
    "A": ["A", "Bm", "C#m", "D", "E", "F#m", "G#o"],
    "A#": ["Bb", "Cm", "Dm", "Eb", "F", "Gm", "Ao"],
    "B": ["B", "C#m", "D#m", "E", "F#", "G#m", "A#o"]
}


class Chord:
    # notes are integers
    def __init__(self, a, b, c, type):
        self.first_note = a
        self.second_note = b
        self.third_note = c
        self.time = 0
        # 11 - Major
        # 12 - Minor
        # 2111 - first inversions of major
        # 2112 - first inversions of minor
        # 2211 - second inversions of major
        # 2212 - second inversions of minor
        # 3 - diminished chords (DIM)
        # 4 - suspended second chords (SUS2)
        # 5 - suspended fourth chords (SUS4)
        self.type = type  # 0 - empty chord
        self.main_note_str = pretty_midi.note_number_to_name(a)

    # returns main note (A,B,C...) in integer
    def get_main_note(self):
        return self.first_note % 12

    def has_note(self, note):
        return (self.first_note % 12 == note) or (self.second_note % 12 == note) or (self.third_note % 12 == note)

    def set_time(self, time):
        self.time = time  # end time of the lowest note

    def get_octave(self):
        return self.first_note // 12

    def equal_to(self, chord):
        return (self.first_note == chord.first_note) and (self.second_note == chord.second_note) and (
                self.third_note == chord.third_note)

    def main_note_name(self):
        return pretty_midi.note_number_to_name(self.get_main_note())[:-1]


# function from open-source project "https://www.programcreek.com/python/?project_name=jongwook%2Fonsets-and-frames"
def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = path

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity,
                         sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


mid = MidiFile('input3.mid', clip=True)
lower_bound = parse_midi(mid)[0][2] - 26  # lower bound for range of chord's main note
upper_bound = lower_bound + 24  # upper bound is 2 octaves higher, so all chords are generated in 2 octaves boundary


# returns note of the melody at the given time
def get_note_at_time(t):
    for i in parse_midi(mid):
        if round(i[0], 2) <= t <= round(i[1], 2):
            return int(i[2])
    return -1


# returns the melody tempo
def tempo(m):
    for track in m.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo / 1000


def get_random_chord() -> Chord:
    n = np.random.randint(1, 6)  # randomly choose type of chords:
    # 0 - empty chord
    # 1 - Major and Minor
    # 2 - first and second inversions of major and minor triads
    # 3 - diminished chords (DIM)
    # 4 - suspended second chords (SUS2)
    # 5 - suspended fourth chords (SUS4)

    if n == 1:
        m = np.random.randint(0, 2)  # random between major (0) and minor(1)
        if m == 0:  # major
            note = np.random.randint(lower_bound, upper_bound)  # creates chord based on random note
            chord = Chord(note, note + 4, note + 7, 11)
            return chord
        else:  # minor
            note = np.random.randint(lower_bound, upper_bound)  # creates chord based on random note
            chord = Chord(note, note + 3, note + 7, 12)
            return chord
    elif n == 2:
        inv = np.random.randint(1, 3)  # choose type of inversion
        m = np.random.randint(0, 2)  # random between major (0) and minor(1)
        if inv == 1:
            if m == 0:  # major
                note = np.random.randint(lower_bound, upper_bound)
                chord = Chord(note, note + 3, note + 9, 2111)  # main note one octave above
                return chord
            else:  # minor
                note = np.random.randint(lower_bound, upper_bound)
                chord = Chord(note, note + 4, note + 9, 2112)  # main note one octave above
                return chord
        else:
            if m == 0:  # major
                note = np.random.randint(lower_bound, upper_bound)
                chord = Chord(note, note + 3, note + 9, 2211)  # major second inversion (-7 to each note)
                return chord
            else:  # minor
                note = np.random.randint(lower_bound, upper_bound)
                chord = Chord(note, note + 5, note + 8, 2212)  # main note and middle note are one octave above
                return chord
    elif n == 3:
        note = np.random.randint(lower_bound, upper_bound)
        chord = Chord(note, note + 3, note + 6, 3)
        return chord
    elif n == 4:
        note = np.random.randint(lower_bound, upper_bound)
        chord = Chord(note, note + 2, note + 7, 4)
        return chord
    elif n == 5:
        note = np.random.randint(lower_bound, upper_bound)
        chord = Chord(note, note + 5, note + 7, 5)
        return chord
    elif n == 0:
        chord = Chord(0, 0, 0, 0)
        return chord


# return list of 8 chords
def get_individual() -> List[Chord]:
    accompaniment = []
    for i in range(chord_numbers):
        chord = get_random_chord()
        chord.set_time(chord_duration)  # duration time
        accompaniment.append(chord)
    return accompaniment


# returns the fitness of a given sequence
def get_fitness(individual: List[Chord]) -> float:
    # fitness of one individual

    # give points for:
    # - chord contains note which is playing now
    # - chord does not overlaps the melody
    # - decrease the fitness by the mean deviation from mean note
    fitness = 0
    i = 0  # range(0,15)
    for chord in individual:
        t = (i * chord_duration / 1000)  # calculating the current time of the midi
        if (chord.get_main_note() == get_note_at_time(t) % 12) and (chord.first_note < get_note_at_time(t)):
            fitness += 25
        if chord.second_note % 12 == (get_note_at_time(t) % 12) or chord.third_note % 12 == (
                get_note_at_time(t) % 12):
            fitness += 15
        if not chord.has_note(get_note_at_time(t) % 12):
            fitness -= 30
        i += 1

    # main note of the first chord in each individual defines the tone of the accompaniment
    # main note of the first chord should correspond to first note of the melody
    # individual gets points for accordance with tone
    key_chord = individual[0]
    key_chord_str = pretty_midi.note_number_to_name(key_chord.first_note)
    key_chord_str = key_chord_str[:-1]
    if key_chord.get_main_note() == get_note_at_time(0) % 12:
        fitness += 25

    for chord in individual:
        # define minor keys first
        # if chord in list of corresponding to key chords
        cur_chord_str = pretty_midi.note_number_to_name(key_chord.first_note)
        cur_chord_str = cur_chord_str[:-1]
        if key_chord.type == 11:  # major keys
            if cur_chord_str in major_keys.get(key_chord_str):
                fitness += 30
            else:
                fitness -= 0
        elif key_chord.type == 12:  # minor keys
            if cur_chord_str in minor_keys.get(key_chord_str):
                fitness += 30
            else:
                fitness -= 0
        else:
            fitness -= 50
    # calc mean note
    mean_deviation = calc_mean_dev(individual)
    max_deviation = calc_deviation(individual)

    if individual[0].equal_to(individual[-1]):
        fitness += 100

    if mean_deviation >= 6:
        fitness -= mean_deviation * 5
        fitness -= max_deviation * 6

    return fitness


# returns mean deviation of chords in one individual
def calc_mean_dev(individual):
    sum = 0
    for i in individual:
        sum += i.first_note
    mean_note = sum / chord_numbers
    mean_deviation = 0
    for i in individual:
        mean_deviation += (i.first_note - mean_note) ** 2
    return mean_deviation ** 0.5


# returns max deviation of chords in one individual
def calc_deviation(individual):
    mean_deviation = calc_mean_dev(individual)
    max_deviation = 0
    for i in individual:
        max_deviation = max(max_deviation,
                            max((abs(mean_deviation - i.first_note), abs(mean_deviation - i.third_note))))
    return max_deviation


# return list of chords
def get_population(population_size: int) -> list[list[Chord]]:
    return [get_individual() for i in range(population_size)]


# returns list of individual's fitness and average fitness of the population
def population_fitness(population: List[List[Chord]]) -> Tuple[List[float], float]:
    # fitness = [get_fitness(individual) for individual in population]
    fitness = []
    for individual in population:
        fitness.append(get_fitness(individual))
    return fitness, float(np.mean(fitness))


# returns index of a selected parent
def roulette_wheel_select(fitness: List[float]) -> int:
    # you may use np.random.choice
    return fitness.index(np.random.choice(fitness))


def crossover(population: List[List[Chord]], fitness: List[float], size: int) -> List[List[Chord]]:
    # selects two parents to generate offspring
    # this process continues "size" times
    # returns list of offsprings
    offsprings = []
    for i in range(size):
        n1 = np.random.randint(len(population))
        n2 = np.random.randint(len(population))
        offsprings.append(list(population[n1]))
        offsprings.append(list(population[n2]))
    return offsprings


def mutate(offsprings: List[List[Chord]]) -> List[List[Chord]]:
    # mutate by changing several the most deflecting chord on a random chord in each offspring
    max_index = 0
    max_dev = 0
    for individual in offsprings:
        for a in range(chord_numbers // 4):
            sumg = 0
            for h in individual:
                sumg += h.first_note
            mean_note = sumg / chord_numbers
            g = 0
            for chord in individual:
                if max(abs(chord.first_note - mean_note), abs(chord.third_note - mean_note)) > max_dev:
                    max_dev = max(abs(chord.first_note - mean_note), abs(chord.third_note - mean_note))
                    max_index = g
                g += 1
            individual[max_index] = get_random_chord()
            g = 0
            max_dev = 0
            max_index = 0
    return offsprings


# some melodies have delay before the start playing
def start_time(m):
    if (parse_midi(m)[0][0]) == 0:
        return 0
    else:
        return m.ticks_per_beat / 2


def replace_parents(population: List[List[Chord]], population_fitness_list: List[float], offsprings: List[List[Chord]],
                    offsprings_fitness: List[float], size: int) -> List[List[Chord]]:
    # replace "size" number of least fit population members
    # with most fit "size" offsprings
    # returns new population

    sort_index = np.argsort(population_fitness_list)
    sort_index = sort_index[::-1]
    population_sorted = []
    for g in sort_index:
        population_sorted.append(population[g])

    sort_index = np.argsort(offsprings_fitness)
    offsprings_sorted = []
    sort_index = sort_index[::-1]
    for g in sort_index:
        offsprings_sorted.append(offsprings[g])

    parents = population_sorted[:-size]  #
    offsprings = offsprings_sorted[:size]
    res = parents + offsprings

    res_fitness = population_fitness(res)
    sort_index = np.argsort(res_fitness[0])
    sort_index = sort_index[::-1]
    t = []
    for g in sort_index:
        t.append(res[g])
    return t


def evolution(generations: int, population_size: int):
    population = get_population(population_size)

    for generation in range(generations):
        fitness, avg_fitness = population_fitness(population)
        offsprings = crossover(population, fitness, 30)
        offsprings = mutate(offsprings)
        offsprings_fitness, offsprings_fitness_avg = population_fitness(offsprings)
        fitness, avg_fitness = population_fitness(population)
        population = replace_parents(population, fitness, offsprings, offsprings_fitness, 10)
    return population


chord_duration = mid.ticks_per_beat * 2  # chord duration is chosen as 4 quarters
delta = (parse_midi(mid)[-1][1] - parse_midi(mid)[0][1]) * 1000  # melody duration
chord_numbers = round(delta / tempo(mid) * 0.5)

generationss = 20  # this value obtained
populations = evolution(generationss, population_size=40)

mid2 = MidiFile()
mid2.add_track()
vel = int(parse_midi(mid)[0][-1])  # volume of chords track

for i in range(chord_numbers):
    if i == 0:
        mid2.tracks[0].append(
            Message('note_on', note=populations[0][i].first_note, velocity=vel, time=round(start_time(mid))))
    else:
        mid2.tracks[0].append(Message('note_on', note=populations[0][i].first_note, velocity=vel, time=0))
    mid2.tracks[0].append(Message('note_on', note=populations[0][i].second_note, velocity=vel, time=0))
    mid2.tracks[0].append(Message('note_on', note=populations[0][i].third_note, velocity=vel, time=0))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].first_note, velocity=vel, time=chord_duration))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].second_note, velocity=vel, time=0))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].third_note, velocity=vel, time=0))

merged_mid = MidiFile()
merged_mid.ticks_per_beat = mid.ticks_per_beat
merged_mid.tracks = mid.tracks + mid2.tracks
merged_mid.save('output3.mid')
