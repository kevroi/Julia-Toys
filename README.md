After installing Julia, run this command to set its PATH
```
ln -s /Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```
Use this cmmand to see what's the name of the version of Julia you have installed in Applications (MacOS)
```
ls -a /Applications/
```

## Getting Started
As an example, to run the artificial neural neural network that classifies handwritten digits from MNIST (the *Hello World* of machine learning) in your terminal:

```sh-session
$ cd MNIST
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.3 (2022-11-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |
$ julia> ]
$ (@v1.8) pkg> activate .
$ julia> include("ann.jl")
```

Pressing the `]` key shows you your package REPL, and `activate .` activates a virtual env like in python.


## Useful Functions
`readdir()` on Julia REPL shows available directories.
`cd()` to go to where you want to be.
`exit()` to leave the Julia REPL.

Pressing the `]` key shows you your package REPL, and `activate .` activates a virtual env like in python.

## Tips
ML datasets are downloaded locally. MNIST was saved to `/Users/kevinroice/.julia/datadeps/MNIST` on my machine.