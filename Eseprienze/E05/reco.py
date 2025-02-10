import numpy as np
import pandas as pd

class Hit:
    '''
    Classe che descrive una singola rivelazione in base al modulo, al sensore e 
    all'istante di tempo di rivelazione a partire dall'istante 0.
    
    parametri
    ------------
        mod_id: id del modulo di rilevamento
        det_id: id del sensore di rilevamento
        hit_time: istante di tempo di rilevazione
    '''
    
    def __init__(self, mod, det, time):
        self.mod_id = mod
        self.det_id = det
        self.hit_time = time
        
    def __str__(self):
        return f"mod_id: {self.mod_id} \t det_id: {self.det_id} \t hit_time: {self.hit_time}"
    
    # overloading ordinabilità e operazioni Hit
    def __lt__(self, other):
        return self.hit_time < other.hit_time
    
    def __gt__(self, other):
        return self.hit_time > other.hit_time
    
    def __le__(self, other):
        return self.hit_time <= other.hit_time
    
    def __ge__(self, other):
        return self.hit_time >= other.hit_time
    
    def __add__(self, other):
        return self.hit_time + other.hit_time
    
    def __sub__(self, other):
        return self.hit_time - other.hit_time
    
    # ordinamento per id modulo
    def mod_lt(self, other):
        return self.mod_id < other.mod_id
    
    def mod_gt(self, other):
        return self.mod_id > other.mod_id
    
    def mod_le(self, other):
        return self.mod_id <= other.mod_id
    
    def mod_ge(self, other):
        return self.mod_id >= other.mod_id
    
    # ordinamento per id detector
    def det_lt(self, other):
        return self.mod_id < other.mod_id
    
    def det_gt(self, other):
        return self.mod_id > other.mod_id
    
    def det_le(self, other):
        return self.mod_id <= other.mod_id
    
    def det_ge(self, other):
        return self.mod_id >= other.mod_id
    

class Event:
    '''
    Classe che descrive un evento di rivelazione 
    
    parametri
    ------------
        hit_num: numero di hit dell'evento
        first_hit: primo hit in ordine cronologico dell'evento
        last_hit: ultimo hit in ordine cronologico dell'evento
        time_lapse: finestra temporale dell'evento definita come last_hit - first_hit
        hit_list:array degli hit dell'evento
    '''
    hit_num = 0
    first_hit = 0
    last_hit = 0
    time_lapse = 0
    hit_list = np.array([], dtype=Hit)
    
    def __init__(self, hitList: np.ndarray):
        self.hit_list = hitList
        self.hit_num = len(hitList)
        self.first_hit = hitList[0].hit_time
        self.last_hit = hitList[-1].hit_time
        self.time_lapse = self.last_hit - self.first_hit
    
    def print(self):
        print(f"Numero di hit dell'evento: {self.hit_num}")
        print(f"Tempo del primo hit dell'evento: {self.first_hit}")
        print(f"Tempo dell'ultimo hit dell'evento: {self.last_hit}")
        print(f"Durata dell'evento: {self.time_lapse}")
        print("mod_id\t\tdet_id\t\thit_time")
        for i in self.hit_list:
            print(f"{i.mod_id}\t\t{i.det_id}\t\t{i.hit_time}")
            
    def __iter__(self):
        return iter(self.hit_list)
        
    

def _hit_gen_csv(df: pd.DataFrame):
    for i in df.values:
        yield Hit(*i)

def array_hit(csv: str) -> np.ndarray:
    df = pd.read_csv(csv)
    obj = np.array([i for i in _hit_gen_csv(df)])
    return obj

def _event_gen_array(obj):
    arr = np.concatenate(([0], np.where(obj[1:] - obj[:-1] > 1000)[0], [len(obj)]))
    print(arr)
    for i in range(len(arr)-1):
        yield obj[arr[i]:arr[i+1]]

def array_event(hit):
    if type(hit) == np.ndarray:
        return [Event(i) for i in _event_gen_array(hit)]
