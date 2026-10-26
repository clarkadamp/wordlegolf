{
  description = "Wordlegolf letter generator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in

        {
          default = pkgs.writeShellApplication {
            name = "wordle-golf";
            text = ''
              STATS_FILE=${./data/WordleGolfBetterProbs.csv} ${
                pkgs.writers.writePython3Bin "letters.py" {
                  flakeIgnore = [
                    "E226"
                    "E265"
                    "E501"
                    "E203"
                  ];
                } (builtins.readFile ./letters.py)
              }/bin/letters.py "$@"
            '';
          };
        }
      );
    };
}
