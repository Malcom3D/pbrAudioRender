# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

__version__ = "0.9.1.3"
__author__ = "Malcom3D"
__description__ = "3D acoustic rendering engine"

import os, sys
from .core.entity_manager import EntityManager
from .core.acoustic_engine import AcousticEngine
from .core.audio_recorder import AudioRecorder

class pbrAudioRender:
    """Main class for running 3D acoustic simulations"""
    def __init__(self, config_file: str):
        self.em = EntityManager(config_file)
        self.ac = AcousticEngine(self.em)
        self.ar = AudioRecorder(self.em)
#        self.am = AmbisonicRender(self.em)

    def start(self):
        frames = self.em.get('frames')
        while frames.get() < frames.get_limit():
            print(f"Processing frame: {frames.get()}")
            self.ac.update()
            self.ar.update()
            frames.next()

#        self.am.render()
