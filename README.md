After installing Julia, run this command to set its PATH
```
ln -s /Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```
Use this cmmand to see what's the name of the version of Julia you have installed in Applications (MacOS)
```
ls -a /Applications/
```

## Useful Functions
Use `readdir()` on Julia REPL to see available directories, and `cd()` to go to where you want to be.

Pressing the `]` key shows you your package REPL, and `activate .` activates a virtual env like in python.

## Tips
ML datasets are downloaded locally. MNIST was saved to `/Users/kevinroice/.julia/datadeps/MNIST` on my machine.