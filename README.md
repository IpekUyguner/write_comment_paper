# write_comment_paper

Pytorch loader for the project http://groups.inf.ed.ac.uk/cup/comment-locator/ 
------------------------------------------------------------------------------

-Download the embeddings from the above link under "data/code-big-vectors-negative300.bin"

-Install the missing packages:

  python 3.6.0 

  gensim==3.8.0

  numpy

  smart-open (for gensim)

  torch==1.4.0

  torchvision==0.5.0



-Change CODE_EMBEDDINGS_DATA in pytorch_data_reader.py to the directory of "code-big-vectors-negative300.bin"
-Run pytorch_data_reader.py by entering data.txt path as input
-Output demonstrates the dataloader object
