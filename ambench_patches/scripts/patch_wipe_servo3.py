"""WipeWindow: the approach itself slams the sponge into the window (41 N at the 'close' pose, before the servo can act):
the PID overshoots when decelerating 25 cm from the window. Stand off further (close 0.30, contact 0.27) and let the servo
walk in; retreat much faster than advance so any overshoot is relieved within a few steps."""
import pathlib
p = pathlib.Path("/public/home/liaodl/ambench/src/ambench/source/ambench/ambench/policies/scripted/wipe_window.py")
s = p.read_text()
pairs = [
 ("            close[0] = dot_pos[0] - 0.25\n", "            close[0] = dot_pos[0] - float(os.environ.get(\"AMBENCH_WIPE_CLOSE\", \"0.30\"))  # stand-off: PID overshoot on approach must not reach the window\n"),
 ('contact[0] = dot_pos[0] - float(os.environ.get("AMBENCH_WIPE_CONTACT", "0.24"))', 'contact[0] = dot_pos[0] - float(os.environ.get("AMBENCH_WIPE_CONTACT", "0.27"))'),
 ('''            if f < f_des - 1.0:
                dx += step
            elif f > f_des + 1.0:
                dx -= step''',
  '''            if f < f_des - 1.0:
                dx += step                      # advance slowly into contact
            elif f > 4.0 * f_des:
                dx -= 8.0 * step                # hard overshoot: back off fast
            elif f > f_des + 1.0:
                dx -= 4.0 * step'''),
]
for old, new in pairs:
    n = s.count(old); assert n == 1, (n, old[:60]); s = s.replace(old, new)
p.write_text(s); print("patched: close 0.30, contact 0.27, asymmetric servo")
