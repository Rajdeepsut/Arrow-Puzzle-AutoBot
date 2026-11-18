package com.arrowpuzzle.autobot.test;

import com.arrowpuzzle.autobot.core.Frame;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * Loads the captured frames used by the regression tests.
 *
 * <p>The fixtures are a tiny little-endian header (width, height) followed by
 * zlib-compressed raw ARGB pixels.  That keeps the tests on plain Java without
 * pulling in AWT, which is not on the Android unit-test boot classpath.</p>
 */
final class TestFrames {
    private TestFrames() {
    }

    static Frame load(InputStream stream) throws IOException {
        try (DataInputStream input = new DataInputStream(stream)) {
            int width = Integer.reverseBytes(input.readInt());
            int height = Integer.reverseBytes(input.readInt());
            byte[] pixels = inflate(input.readAllBytes(), width * height * 4);
            Frame frame = new Frame(width, height);
            ByteBuffer.wrap(pixels).order(ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer().get(frame.pixels);
            return frame;
        }
    }

    private static byte[] inflate(byte[] compressed, int expected) throws IOException {
        Inflater inflater = new Inflater();
        inflater.setInput(compressed);
        byte[] out = new byte[expected];
        int offset = 0;
        try {
            while (offset < expected) {
                int count = inflater.inflate(out, offset, expected - offset);
                if (count == 0) {
                    if (inflater.finished() || inflater.needsInput()
                            || inflater.needsDictionary()) {
                        break;
                    }
                }
                offset += count;
            }
        } catch (DataFormatException exc) {
            throw new IOException("bad zlib fixture", exc);
        } finally {
            inflater.end();
        }
        if (offset != expected) {
            throw new IOException("frame fixture is " + offset + "/" + expected + " bytes");
        }
        return out;
    }
}
