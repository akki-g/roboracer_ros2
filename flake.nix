{
  nixConfig = {
    extra-substituters = [ "http://100.88.115.111:8080/c/admin/ros-test" ];
    extra-trusted-public-keys = [ "ros-test:jqJlOuADuaw5ElHZRYPEX/NZpyj5u3c2ydIPf0A5OaM=" ];
  };
  inputs = {
    flake-utils = { url = "github:numtide/flake-utils"; };
    # Nix Ros Overlay
    nix-ros-overlay = { url = "github:lopsided98/nix-ros-overlay/master"; };
    nixpkgs = { follows = "nix-ros-overlay/nixpkgs"; };
  };
  outputs =
    { nixpkgs, flake-utils, nix-ros-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = (import nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        });
      in {
        devShells.default = with pkgs;
          mkShell.override { stdenv = pkgs.clangStdenv; } {
            # TORCH_LIBRARIES = "${torch.packages.x86_64-linux.libtorch.out}/lib";
            CMAKE_MODULE_PATH = "${pkgs.opencv.out}/lib/cmake/opencv4";

            # shellHook = ''
            #   # Bind the host WSL driver directories back into the linker path
            #   export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
              
            #   # (Optional) If you are using standard Ubuntu/Debian WSL instead of NixOS-WSL,
            #   # ensure your DISPLAY and Wayland variables are preserved:
            #   export DISPLAY=''${DISPLAY:-:0}
            # '';

            packages = [
              # Generic DevTools
              # clang-tools must be first before clang
              pkgs.clang-tools
              pkgs.cmake
              pkgs.glib
              pkgs.gtk2
              pkgs.clang
              pkgs.nixfmt
              (pkgs.opencv.override {
                enableJPEG = true;
                enableFfmpeg = true;
              })
              # libGL
              # libglvnd
              # xorg.libX11

              # ROS Packages
              pkgs.colcon
              (with pkgs.rosPackages.humble;
                buildEnv {
                  paths = [
                    ros-core
                    image-transport
                    sensor-msgs
                    cv-bridge
                    ament-cmake
                    ament-cmake-core
                    ament-cmake-ros
                    ament-cmake-auto
                    foxglove-bridge
                    ackermann-msgs
                    diagnostic-updater
                    python-cmake-module
                    rviz2

                    # ... other ROS packages
                  ];
                })
            ];
          };
      });
}
