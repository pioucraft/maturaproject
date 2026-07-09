{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    packages = with pkgs; [
        (texlive.withPackages (ps: with ps; [
            scheme-medium
            pgfplots
        ]))
    ];
}
