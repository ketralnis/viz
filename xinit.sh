#!/bin/sh

setterm -blank 0
setterm -powersave off
setterm -powerdown
xset s off
xset -dpms
xset s noblank

cd /home/pi/viz
sudo -u pi /home/pi/.cargo/bin/cargo build --release
exec `pwd`/target/release/viz
