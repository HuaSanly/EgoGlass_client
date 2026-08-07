#!/usr/bin/env bash
set -euo pipefail

source_directory=${1:-/home/nvidia/egoglass/tools/basalt-src}
build_directory=${2:-/home/nvidia/egoglass/tools/basalt-build}
patch_directory=${3:?patch directory is required}
basalt_revision=${4:-0f3b2b52c807f70ff4e2973ce253c73329eea7bc}
proxy_url=${5:--}
cmake_version=3.31.10
tools_directory=$(dirname "$source_directory")
cmake_directory="$tools_directory/cmake-$cmake_version-linux-x86_64"
download_directory="$HOME/.cache/egoglass/downloads"

if [[ "$proxy_url" != "-" ]]; then
    export HTTP_PROXY="$proxy_url"
    export HTTPS_PROXY="$proxy_url"
    export http_proxy="$proxy_url"
    export https_proxy="$proxy_url"
fi

build_jobs=${EGOGLASS_BASALT_BUILD_JOBS:-4}
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "EGOGLASS_BASALT_BUILD_JOBS must be a positive integer." >&2
    exit 64
fi
export VCPKG_MAX_CONCURRENCY="$build_jobs"

if ! sudo -n true >/dev/null 2>&1; then
    echo "The WSL user needs passwordless sudo for automated dependency setup." >&2
    exit 77
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    libtool \
    libx11-dev \
    ninja-build \
    pkg-config \
    rsync \
    tar \
    unzip \
    zip

cmake_executable=$(command -v cmake)
system_cmake_version=$($cmake_executable --version | awk 'NR == 1 {print $3}')
if ! dpkg --compare-versions "$system_cmake_version" ge 3.24; then
    if [[ "$(uname -m)" != "x86_64" ]]; then
        echo "The pinned CMake bootstrap currently supports WSL x86_64 only." >&2
        exit 69
    fi
    mkdir -p "$tools_directory" "$download_directory"
    cmake_archive="$download_directory/cmake-$cmake_version-linux-x86_64.tar.gz"
    if [[ ! -f "$cmake_archive" ]]; then
        curl --fail --location --retry 3 \
            "https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.tar.gz" \
            --output "$cmake_archive"
    fi
    if [[ ! -x "$cmake_directory/bin/cmake" ]]; then
        tar -xzf "$cmake_archive" -C "$tools_directory"
    fi
    cmake_executable="$cmake_directory/bin/cmake"
fi

mkdir -p "$tools_directory" "$build_directory"
if [[ ! -d "$source_directory/.git" ]]; then
    mkdir -p "$source_directory"
    git -C "$source_directory" init
    git -C "$source_directory" remote add origin \
        https://github.com/VladyslavUsenko/basalt.git
    git -C "$source_directory" fetch --depth 1 origin "$basalt_revision"
    git -C "$source_directory" checkout --detach FETCH_HEAD
    git -C "$source_directory" submodule update --init --recursive
fi

vcpkg_directory="$source_directory/thirdparty/vcpkg"
if [[ "$(git -C "$vcpkg_directory" rev-parse --is-shallow-repository)" == "true" ]]; then
    git -C "$vcpkg_directory" fetch --unshallow origin
fi
vcpkg_baseline=$(sed -n \
    's/.*"baseline"[[:space:]]*:[[:space:]]*"\([0-9a-f]\{40\}\)".*/\1/p' \
    "$source_directory/vcpkg-configuration.json")
if [[ ! "$vcpkg_baseline" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Basalt vcpkg baseline is missing or invalid." >&2
    exit 65
fi
git -C "$vcpkg_directory" cat-file -e "$vcpkg_baseline^{commit}"

git -C "$source_directory" fetch origin "$basalt_revision"
if [[ -n "$(git -C "$source_directory" status --short)" ]]; then
    current_revision=$(git -C "$source_directory" rev-parse HEAD)
    if [[ "$current_revision" != "$basalt_revision" ]]; then
        echo "Patched Basalt source is not at the requested revision." >&2
        echo "expected: $basalt_revision" >&2
        echo "actual:   $current_revision" >&2
        exit 65
    fi
    for patch in "$patch_directory"/*.patch; do
        if ! git -C "$source_directory" apply --unidiff-zero --reverse --check "$patch" >/dev/null 2>&1; then
            echo "Basalt source contains changes unrelated to the EgoGlass patch set." >&2
            git -C "$source_directory" status --short >&2
            exit 65
        fi
    done
else
    git -C "$source_directory" checkout --detach "$basalt_revision"
    git -C "$source_directory" submodule update --init --recursive
    for patch in "$patch_directory"/*.patch; do
        git -C "$source_directory" apply --unidiff-zero --check "$patch"
        git -C "$source_directory" apply --unidiff-zero "$patch"
    done
fi
if [[ "$(git -C "$source_directory" rev-parse HEAD)" != "$basalt_revision" ]]; then
    echo "Basalt source revision verification failed." >&2
    exit 65
fi

configure_attempt=1
while ! "$cmake_executable" -S "$source_directory" -B "$build_directory" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$source_directory/thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake" \
    -DCMAKE_CXX_FLAGS=-Wno-error=unused-variable \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON; do
    if [[ $configure_attempt -ge 3 ]]; then
        echo "Basalt dependency configuration failed after $configure_attempt attempts." >&2
        exit 1
    fi
    configure_attempt=$((configure_attempt + 1))
    echo "Retrying Basalt dependency configuration ($configure_attempt/3)." >&2
    sleep 3
done
"$cmake_executable" --build "$build_directory" --parallel "$build_jobs" \
    --target basalt_vio basalt_calibrate basalt_calibrate_imu

"$build_directory/basalt_vio" --help >/dev/null
"$build_directory/basalt_calibrate" --help >/dev/null
"$build_directory/basalt_calibrate_imu" --help >/dev/null

printf '{\n  "basalt_revision": "%s",\n  "build_type": "Release",\n  "cmake_version": "%s"\n}\n' \
    "$basalt_revision" "$($cmake_executable --version | awk 'NR == 1 {print $3}')" \
    > "$build_directory/egoglass-build.json"
echo "Basalt WSL tools are ready in $build_directory"
