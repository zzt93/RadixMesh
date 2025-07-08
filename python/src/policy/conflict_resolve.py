class NodeRankConflictResolver:
    @staticmethod
    def keep(now_value_rank: int, new_value_rank) -> bool:
        if now_value_rank <= new_value_rank:
            return True
        return False
