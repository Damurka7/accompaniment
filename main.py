import mido
from mido import MidiFile, Message
import numpy as np
import time
from typing import Union, List, Tuple

# divide song on pieces of 1/4 and create sequences of random chords
# each sequence is a list of chords, each chord continues for 1/4
# each sequence is an individual
# each sequence may have chords distributed along whole octaves
# mutation changes several random chords in an individual


chord_duration = 960
chord_numbers = 8


class Chord:
    # notes are integers
    def __init__(self, a, b, c, chord_type):
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
        self.chord_type = chord_type

    # returns main note (A,B,C...) in integer
    def get_main_note(self):
        return self.first_note % 12

    def has_note(self, note):
        return (self.first_note == note) or (self.second_note == note) or (self.third_note == note)

    def set_time(self, time):
        self.time = time  # end time of the lowest note

    def get_octave(self):
        return self.first_note // 12

    def equal_to(self, chord):
        return (self.first_note == chord.first_note) and (self.second_note == chord.second_note) and (
                    self.third_note == chord.third_note)


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


def get_note_at_time(t):
    for i in parse_midi(mid):
        if round(i[0], 2) <= t <= round(i[1], 2):
            return int(i[2])


def fun(name):
    msgs = MidiFile(name, clip=True).tracks[0]
    del msgs[0]
    del msgs[len(msgs) - 1]
    return msgs


lower_bound = 36
upper_bound = 59

def get_random_chord() -> Chord:
    n = np.random.randint(1, 6)  # randomly choose type of chords:
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
                chord = Chord(note, note + 5, note + 11, 2211)  # major second inversion (-7 to each note)
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


# return list of 8 chords
def get_individual() -> List[Chord]:
    accompaniment = []
    for i in range(chord_numbers):
        chord = get_random_chord()
        chord.set_time(chord_duration)  # duration time
        accompaniment.append(chord)
    return accompaniment


# should return the fitness of a given sequence
def get_fitness(individual: List[Chord]) -> float:
    # fitness of one individual

    # give points for:
    # - chord contains note which is playing now
    # - chord does not overlaps the melody
    # - main chord not one-two octaves lower
    # - decrease the fitness by the mean deviation from mean note
    fitness = 0
    i = 0  # range(0,15)
    for chord in individual:
        t = (i * chord_duration / 1000)  # calculating the current time of the midi
        if (chord.get_main_note() == get_note_at_time(t) % 12) and (chord.first_note <= get_note_at_time(t)):
            fitness += 20
        if chord.second_note % 12 == (get_note_at_time(t) % 12) or chord.third_note % 12 == (
                get_note_at_time(t) % 12):
            fitness += 10

        if chord.has_note(get_note_at_time(t)):
            fitness += 10

        # if chord.first_note - 12 == get_note_at_time(t):
        #     fitness += 15
        # elif chord.first_note - 24 == get_note_at_time(t):
        #     fitness += 5
        # else:
        #     fitness -= 10

        i += 1

    # first and last chord are the same and are in the tonic
    # if individual[0].equal_to(individual[-1]) :
    #     fitness += 40

    # calc mean note
    sum = 0
    for i in individual:
        sum += i.first_note
    mean_note = sum / chord_numbers
    mean_deviation = 0
    for i in individual:
        mean_deviation += (i.first_note - mean_note) ** 2

    mean_deviation = mean_deviation ** 0.5

    max_deviation = 0
    for i in individual:
        max_deviation = max(max_deviation, max((abs(mean_deviation-i.first_note), abs(mean_deviation-i.third_note))))

    fitness -= mean_deviation * 6
    fitness -= max_deviation * 4
    print("mean_deviation: ", mean_deviation)
    print("max_deviation: ", max_deviation)
    print("fitness: ", fitness)

    return fitness


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
        # num1 = np.random.choice(population)
        # num2 = np.random.choice(population)
        # offsprings.append(num2)
        # offsprings.append(num1)

        n1 = np.random.randint(len(population))
        n2 = np.random.randint(len(population))

        offsprings.append(population[n1])
        offsprings.append(population[n2])
    # print("type of offspring ", type(offsprings))
    return offsprings


def mutate(offsprings: List[List[Chord]]) -> List[List[Chord]]:
    # mutates by changing all chords in an individual

    # for j in range(2):
    #     i = np.random.randint(len(offsprings) - 1)
    #     # change all of chords for a given offspring
    #
    #     n = np.random.randint(chord_numbers)
    #     offsprings[i][n] = get_random_chord()

    ind_n = np.random.randint(len(offsprings))
    for g in range(ind_n):
        i = np.random.randint(len(offsprings) - 1)
        offsprings[i] = get_individual()

    # changing several chords in several individuals from offspring list
    # n = 2  # choosing number of individuals
    # for j in range(n):
    #     chord_n = 3  # choosing number of chords
    #     ind_n = np.random.randint(len(offsprings))
    #     for g in range(chord_n):
    #         i = np.random.randint(chord_numbers)  # choosing the positions
    #         offsprings[ind_n][i] = get_random_chord()

    # change several chords in all offsprings
    # for i in offsprings:
    #     # n = np.random.randint(chord_numbers)
    #     for j in range(2):
    #         g = np.random.randint(chord_numbers)
    #         offsprings[j][g] = get_random_chord()

    return offsprings


def replace_parents(population: List[List[Chord]], population_fitness_list: List[float], offsprings: List[List[Chord]],
                    offsprings_fitness: List[float], size: int) -> List[List[Chord]]:
    # replace "size" number of least fit population members
    # with most fit "size" offsprings
    # returns new population
    sort_index = np.argsort(population_fitness_list)
    sort_index = sort_index[::-1]
    population_sorted = population
    c = 0
    for i in sort_index:
        temp_list = population[i]
        population_sorted[c] = temp_list
        c += 1

    # population_sorted = np.take(population, sort_index)
    sort_index = np.argsort(offsprings_fitness)

    offsprings_sorted = offsprings
    sort_index = sort_index[::-1]
    c = 0
    for i in sort_index:
        temp_list = offsprings[i]
        offsprings_sorted[c] = temp_list
        c += 1

    parents = population_sorted[:-size]  #
    offsprings = offsprings_sorted[:size]

    # par_fit = population_fitness(parents)
    # off_fit = population_fitness(offsprings)
    res = parents + offsprings
    # for j in parents:
    #     res.append(i)
    #
    # for j in offsprings:
    #     res.append(i)
    # f = population_fitness(res)
    # print(5)
    return res


def evolution(generations: int, population_size: int):
    population = get_population(population_size)

    for generation in range(generations):
        print(generation)
        fitness, avg_fitness = population_fitness(population)
        offsprings = crossover(population, fitness, 20)
        offsprings = mutate(offsprings)
        offsprings_fitness, offsprings_fitness_avg = population_fitness(offsprings)
        population = replace_parents(population, fitness, offsprings, offsprings_fitness, 10)
        # f = population_fitness(population)

    return population


mid = MidiFile('barbiegirl.mid', clip=True)

generationss = 25
populations = evolution(generationss, population_size=40)

mid2 = MidiFile()
mid2.add_track()

for i in range(chord_numbers):
    mid2.tracks[0].append(Message('note_on', note=populations[0][i].first_note, velocity=127, time=0))
    mid2.tracks[0].append(Message('note_on', note=populations[0][i].second_note, velocity=127, time=0))
    mid2.tracks[0].append(Message('note_on', note=populations[0][i].third_note, velocity=127, time=0))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].first_note, velocity=127, time=chord_duration))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].second_note, velocity=127, time=0))
    mid2.tracks[0].append(Message('note_off', note=populations[0][i].third_note, velocity=127, time=0))

merged_mid = MidiFile()
merged_mid.ticks_per_beat = mid.ticks_per_beat
merged_mid.tracks = mid.tracks + mid2.tracks
merged_mid.save('barbiegirl2.mid')

# print(parse_midi(mid))
