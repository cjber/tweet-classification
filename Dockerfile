FROM docker.io/cjber/cuda

RUN pacman -Syu python-pip pyenv --noconfirm \
    && pip install poetry

WORKDIR /flood
COPY pyproject.toml .python-version ./

RUN yes | pyenv install $(cat .python-version) \
    && poetry install

ENTRYPOINT [ "poetry", "run", "python", "-m", "src.run" ]
