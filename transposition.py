class TranspositionTable:
    def __init__(self, size_mb = 256):
        self.table = {}
        self.lookups = 0
        self.total_hits = 0

    def store(self, zobrist_key, eval, depth, best_line, bound):
        self.table[zobrist_key] = {
            "eval": eval,
            "depth": depth,
            "best": best_line,
            "bound": bound,
            "hits": 0
        }

    def lookup(self, zobrist_key):
        self.lookups += 1
        entry = self.table.get(zobrist_key)
        if entry:
            self.total_hits += 1
            entry["hits"] += 1
        return entry

