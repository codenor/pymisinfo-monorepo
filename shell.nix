let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-25.05.tar.gz") {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      pandas
      matplotlib
      chromadb
      ollama
      rich
      numpy
      scikit-learn
      scipy
      joblib
      matplotlib
      seaborn
      wordcloud
      jupyter
    ]))
  ];
}
