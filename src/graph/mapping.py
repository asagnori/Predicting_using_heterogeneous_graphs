
def get_mapping(df, id_col):
    return {
        old_id: new_id
        for new_id, old_id in enumerate(df[id_col].unique())
    }

def create_edges(df, src_col, dst_col, src_map, dst_map):
    src_idx = [src_map[i] for i in df[src_col]]
    dst_idx = [dst_map[i] for i in df[dst_col]]
    return torch.tensor([src_idx, dst_idx], dtype=torch.long)