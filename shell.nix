{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    python312
    python312Packages.pip

    # SLAM / vision deps
    opencv
    eigen
    ffmpeg

    # build tools
    cmake
    pkg-config
    gcc
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libxcb
      pkgs.libx11
      pkgs.libxext
      pkgs.libxrender
      pkgs.libGL
      pkgs.libGLU
      pkgs.libglvnd
      pkgs.glib.out
      pkgs.libsm
      pkgs.libice
    ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -n "$LIBCUDA_PATH" && -f "$LIBCUDA_PATH" ]]; then
      export LD_PRELOAD="$LIBCUDA_PATH''${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
  '';
}
