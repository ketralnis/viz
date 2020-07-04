#!/bin/sh

setterm -blank 0
setterm -powersave off
setterm -powerdown
xset -dpms

cd /home/pi/viz
sudo -u pi /home/pi/.cargo/bin/cargo build --release
exec `pwd`/target/release/viz
