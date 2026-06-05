from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
cfg: MazeDatasetConfig = MazeDatasetConfig(
    name="test", # name is only for you to keep track of things
    grid_n=5, # number of rows/columns in the lattice
    n_mazes=200, # number of mazes to generate
    maze_ctor=LatticeMazeGenerators.gen_dfs, # algorithm to generate the maze
    maze_ctor_kwargs=dict(do_forks=False), # additional parameters to pass to the maze generation algorithm
)

dataset : MazeDataset = MazeDataset.from_config(cfg)

m = dataset[0]
print(type(m))
# visual representation as ascii art
print(m.as_ascii())

# RGB image, optionally without solution or endpoints, suitable for CNNs
import matplotlib.pyplot as plt
plt.imshow(m.as_pixels())
plt.show()
#plt.savefig("maze.png")

# text format for autoreregressive transformers
from maze_dataset.tokenization import MazeTokenizerModular, TokenizationMode, PromptSequencers
tokens = m.as_tokens(maze_tokenizer=MazeTokenizerModular(
    prompt_sequencer=PromptSequencers.AOTP(), # many options here
))
print(tokens)

# advanced visualization with many features
from maze_dataset.plotting import MazePlot
MazePlot(m).plot()
#plt.savefig("maze_fig.png")
@serializable_dataclass(frozen=True, kw_only=True)
class SolvedMaze(maze_dataset.TargetedLatticeMaze):
	SolvedMaze(
		connection_list: jaxtyping.Bool[ndarray, 'lattice_dim=2 row col'],
		solution: jaxtyping.Int8[ndarray, 'coord row_col=2'],
		generation_meta: dict | None = None,
		start_pos: jaxtyping.Int8[ndarray, 'row_col=2'] | None = None,
		end_pos: jaxtyping.Int8[ndarray, 'row_col=2'] | None = None,
		allow_invalid: bool = False
		)
solution: jaxtyping.Int8[ndarray, 'coord row_col=2']
