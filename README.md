# ORTCustomBeamsearchOpLibrary

### Requirements:
1. A model for internal beam search session.
2. Successful conversion of the model to custom beam search OP using steps in [Convert Script](#convert-script). 
3. 

This is a custom OP to run beam search over a given model. The input to the application would be input ids as shown in the ONNX model.
However, not all the inputs are utilized?? Why do they have to be utilized?

max_length only has to be used for our case, can I make all the inputs optional except for input_ids.  This will save some time to make 
the inputs. 


### Convert script 
Use [create_beam_search.py](create_beam_search.py) to create the custom beam search OP. It takes in the inputs needed to make the path. This doesn't validate that the model is converted, it has to run a session with the model - add this test case. 


