# GenerateYourOwnAnimateCharacter

### Train model

```
sh run.sh
```

### Generate Animate Character

- put your own picture in `./projects/real`

- run `resize.py`

```
cd ./projects
python resize.py
cd ..
```

- generate animate images

```
sh gen.sh
```

- result have been stored in `./projects/latent`
