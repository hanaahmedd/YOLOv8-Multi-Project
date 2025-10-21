# 5) People Entering Counter
elif sel == "people":
    st.header("🚶 People Entering Counter")

    uploaded = st.file_uploader("Upload entrance video", type=["mp4", "avi", "mov"])
    if uploaded:
        # Save uploaded file to a temporary path
        tmp_in = save_uploadedfile_to_tempfile(uploaded)
        cap = cv2.VideoCapture(tmp_in)
        stframe = st.empty()

        # --- Tracking setup ---
        tracker = {}
        next_id = 0
        enter_count = 0

        # --- Helper: Get center ---
        def get_center(xyxy):
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return cx, cy

        # --- Main loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            height, width = frame.shape[:2]

            # Vertical line across whole frame
            line_x = int(width * 0.35)
            line_start = (line_x, 0)
            line_end = (line_x, height)

            # Detect people
            results = models["people"](frame, classes=[0], verbose=False)
            detections = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
            for det in detections:
                x1, y1, x2, y2 = map(int, det[:4])
                cx, cy = get_center((x1, y1, x2, y2))

                # Try to match this detection to existing IDs
                same_person = None
                for pid, prev_center in tracker.items():
                    px, py = prev_center
                    if abs(cx - px) < 40 and abs(cy - py) < 40:
                        same_person = pid
                        tracker[pid] = (cx, cy)
                        break

                # New person
                if same_person is None:
                    tracker[next_id] = (cx, cy)
                    same_person = next_id
                    next_id += 1

                # --- Count every time a red dot comes near the line ---
                if abs(cx - line_x) <= 2:
                    enter_count += 1

                # Draw visuals
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {same_person}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Draw counting line and total
            cv2.line(frame, line_start, line_end, (0, 0, 255), 3)
            cv2.putText(frame, f"Entered: {enter_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # Display in Streamlit
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        st.success(f"✅ Counting finished — Total Entered: {enter_count}")

    if st.button("⬅️ Back to Menu"):
        back_to_menu()
