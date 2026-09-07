# Debian 12 ("bookworm"), the current stable release.
#
# Do not move this back to bullseye. Debian 11 is now oldstable, and Debian
# prunes the actual package files for a release as newer versions supersede
# them. The bullseye image had not been rebuilt since July 2025, so the package
# index baked into it pointed at files that no longer existed on the mirrors and
# every build died on "404 Not Found" while installing ffmpeg. Retrying cannot
# fix a file that has been deleted. Bookworm is rebuilt regularly, so its index
# and its files agree with each other.
FROM python:3.10-slim-bookworm

WORKDIR /app

# ffmpeg is the only system package this service needs. pydub shells out to it
# to read and re-encode uploaded audio, and it needs ffprobe as well, which the
# Debian ffmpeg package provides alongside ffmpeg.
#
# We deliberately do NOT install build-essential or python3-dev any more. Every
# dependency in requirements.txt is available as a prebuilt wheel, so there is
# nothing to compile. Those two packages pulled in hundreds of megabytes of
# compiler toolchain that was never used, which made the image large, the build
# slow, and every deploy that much more likely to trip over a mirror hiccup.
#
# The loop retries a genuinely flaky download rather than failing the deploy. It
# runs apt-get update again on each attempt, because a stale index is the usual
# reason an install fails, and gives up loudly after five tries instead of
# carrying on and shipping an image with no ffmpeg in it.
RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
        echo "ffmpeg install attempt $attempt of 5"; \
        if apt-get update -o Acquire::Retries=5 \
            && apt-get install -y --no-install-recommends -o Acquire::Retries=5 ffmpeg; then \
            echo "ffmpeg installed on attempt $attempt"; \
            break; \
        fi; \
        if [ "$attempt" = "5" ]; then \
            echo "ffmpeg could not be installed after 5 attempts"; \
            exit 1; \
        fi; \
        echo "attempt $attempt failed, retrying"; \
        sleep 5; \
    done; \
    rm -rf /var/lib/apt/lists/*; \
    FFMPEG_V="$(ffmpeg -version)"; \
    echo "$FFMPEG_V" | head -1; \
    FFPROBE_V="$(ffprobe -version)"; \
    echo "$FFPROBE_V" | head -1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deepgram_service:app", "--host", "0.0.0.0", "--port", "8000"]
