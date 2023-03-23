# torchserve_exemple
Simple exemple of torchserve with custom handler


Command line :

To create the models :

```bash
python3 models.py
```


To create the .mar script :

```bash
torch-model-archiver --model-name stardust --version 1.4 --handler handler.py --serialized-file model.pt -f
```

To create launch the server :

```bash
torchserve --model-store models/
```


To test an external API call :

```bash
curl -X POST http://localhost:8080/predictions/stardust -F "inp1=@example_input1.npy" -F "inp2=@example_input2.npy"
```

Configure model serving (change model configuration)  :

```bash
curl -X POST "http://localhost:8081/models?url=stardust.mar&batch_size=8&max_batch_delay=300&min_worker=1"
```


```bash
curl -v -X PUT "http://localhost:8081/models/stardust?min_worker=1"
```


