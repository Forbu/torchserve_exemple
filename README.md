# torchserve_exemple
Simple exemple of torchserve with custom handler


Command line :

To create the models :

python3 models.py



To create the .mar script :

torch-model-archiver --model-name stardust --version 1.4 --handler handler.py --serialized-file model.pt -f


To create launch the server :


torchserve --model-store models/



To test an external API call :


curl -X POST http://localhost:8080/predictions/stardust -F "inp1=@example_input1.npy" -F "inp2=@example_input2.npy"



