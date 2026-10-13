{
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

                    # ... other ROS packages
                  ];
                })
            ];
          };
      });
}
