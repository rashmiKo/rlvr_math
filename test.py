from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=5, # number of rows/columns in the lattice
    n_mazes=4, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset: MazeDataset = MazeDataset.from_config(cfg)

m = dataset[0]

# text format for autoreregressive transformers
from maze_dataset.tokenization import MazeTokenizerModular, TokenizationMode, PromptSequencers
token = m.as_tokens(maze_tokenizer=MazeTokenizerModular(
    prompt_sequencer=PromptSequencers.AOTP(), # many options here
))

print(token)
print(dir(m))

