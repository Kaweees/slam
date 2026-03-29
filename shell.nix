{
  pkgs ? import <nixpkgs> { },
}:

let
  python = pkgs.python312;
in
pkgs.mkShell {
  packages = with pkgs; [
    python
    opencv
    eigen
    ffmpeg
    cyclonedds
  ];

  shellHook = ''
    export TMPDIR=/tmp
    export UV_PYTHON="${python}/bin/python"
    export CYCLONEDDS_HOME="${pkgs.cyclonedds}"
    export CMAKE_PREFIX_PATH="${pkgs.cyclonedds}:$CMAKE_PREFIX_PATH"
    export LD_LIBRARY_PATH="${
      pkgs.lib.makeLibraryPath [
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
      ]
    }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ -n "$LIBCUDA_PATH" && -f "$LIBCUDA_PATH" ]]; then
      export LD_PRELOAD="$LIBCUDA_PATH''${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
  '';
}
