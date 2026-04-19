"""CLI entry point: python3 -m prism compress/decompress/verify"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m prism <command> [args]")
        print("Commands:")
        print("  compress <image> [output.prism]  - Compress image")
        print("  decompress <input.prism> [output.png] - Decompress")
        print("  verify <image> [compressed.prism] - Round-trip verify")
        print("  analyze <image>                   - Analyze prediction quality")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'compress':
        from prism.codec import encode_image
        img_path = sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) > 3 else img_path.rsplit('.', 1)[0] + '.prism'
        encode_image(img_path, out_path)

    elif cmd == 'decompress':
        from prism.codec import decode_image
        in_path = sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) > 3 else in_path.rsplit('.', 1)[0] + '_decoded.png'
        decode_image(in_path, out_path)

    elif cmd == 'verify':
        from prism.codec import encode_image, verify_lossless
        img_path = sys.argv[2]
        prism_path = sys.argv[3] if len(sys.argv) > 3 else '/tmp/verify_test.prism'
        print("=== Step 1: Encode ===")
        encode_image(img_path, prism_path)
        print("\n=== Step 2: Decode + Verify ===")
        verify_lossless(img_path, prism_path)

    elif cmd == 'analyze':
        from prism.analyze import analyze_prediction_quality
        img_path = sys.argv[2]
        crop = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        analyze_prediction_quality(img_path, crop)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == '__main__':
    main()
