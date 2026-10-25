"""Figures 1 and 2: diagrams taken from the AWS Neuron documentation (CC BY-SA 4.0).

Figure 1 puts the Trainium2 chip diagram and the NeuronCore-v3 engine diagram side by side;
Figure 2 is the NKI page's compilation-flow diagram, downscaled. Needs network access.
"""
import io
import os
import urllib.request

from PIL import Image

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/"


def fetch(name):
    with urllib.request.urlopen(DOCS + name, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGBA")


def flat(im):
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3])
    return bg


def arch(out="fig01_trainium2_arch.png", gap=80, pad=30):
    chip = flat(fetch("trainium2.png"))                  # about-neuron/arch/neuron-hardware/trainium2
    core = flat(fetch("nki-trn2-arch-1.png"))            # nki/guides/architecture/trainium2_arch
    core = core.resize((round(core.width * chip.height / core.height), chip.height), Image.LANCZOS)
    canvas = Image.new("RGB", (chip.width + gap + core.width + 2 * pad, chip.height + 2 * pad), "white")
    canvas.paste(chip, (pad, pad))
    canvas.paste(core, (pad + chip.width + gap, pad))
    canvas.save(os.path.join(FIGS, out), optimize=True)


def compile_flow(out="fig02_neuron_compiler_flow.png", width=1600):
    im = flat(fetch("about-nki-3.png"))                   # nki/get-started/about
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(FIGS, out), optimize=True)


if __name__ == "__main__":
    arch()
    compile_flow()
