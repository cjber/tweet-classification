FROM docker.io/cjber/cuda:0.1

ENV PYTHON_VERSION=3.9.7
ENV POETRY_VERSION=1.1.13

ENV USER=user
ENV HOME=/home/$USER

RUN pacman -Syu pyenv blas lapack gcc-fortran --noconfirm

RUN useradd -m user
USER user

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

WORKDIR $HOME/flood
COPY pyproject.toml poetry.lock .envrc ./
COPY src/ ./src
COPY demo_data/ ./data

RUN pyenv install "${PYTHON_VERSION}" \
    && pyenv global "${PYTHON_VERSION}" \
    && pyenv rehash \
    && pip install --no-cache-dir poetry=="${POETRY_VERSION}" \
    && poetry install

ENTRYPOINT [ "poetry", "run", "python", "-m", "src.run" ]
