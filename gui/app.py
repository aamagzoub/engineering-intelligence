import tkinter as tk
from tkinter import ttk

from gui.controller import SimulationController


class WistAILabApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sudanese Wist AI Laboratory")
        self.root.geometry("1150x760")

        self.controller = SimulationController(self)

        self.status_label = None
        self.shota_info_label = None
        self.current_trick_label = None

        self.player_header_labels = [None, None, None, None]
        self.player_bid_labels = [None, None, None, None]
        self.player_card_frames = [None, None, None, None]

        self.played_card_labels = {}

        self._build_layout()

    def _build_layout(self) -> None:
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", padx=12, pady=(8, 4))

        title = ttk.Label(
            top_frame,
            text="Sudanese Wist AI Laboratory",
            font=("Arial", 18, "bold"),
        )
        title.pack()

        self.status_label = ttk.Label(
            top_frame,
            text="Ready",
            font=("Arial", 10),
        )
        self.status_label.pack(pady=2)

        self.shota_info_label = ttk.Label(
            top_frame,
            text="Trump: - | Qabool: - | Shota Bid: - | First Shooter: -",
            font=("Arial", 11, "bold"),
        )
        self.shota_info_label.pack(pady=2)

        table_frame = ttk.LabelFrame(self.root, text="Wist Table")
        table_frame.pack(fill="x", padx=12, pady=6)

        table_frame.columnconfigure(0, weight=1)
        table_frame.columnconfigure(1, weight=1)
        table_frame.columnconfigure(2, weight=1)

        # Player 1 top
        p1 = self._create_player_area(table_frame, 0)
        p1.grid(row=0, column=1, sticky="ew", padx=8, pady=5)

        # Player 4 left
        p4 = self._create_player_area(table_frame, 3)
        p4.grid(row=1, column=0, sticky="ew", padx=8, pady=5)

        # Centre trick
        centre = self._create_centre_area(table_frame)
        centre.grid(row=1, column=1, sticky="nsew", padx=8, pady=5)

        # Player 2 right
        p2 = self._create_player_area(table_frame, 1)
        p2.grid(row=1, column=2, sticky="ew", padx=8, pady=5)

        # Player 3 bottom
        p3 = self._create_player_area(table_frame, 2)
        p3.grid(row=2, column=1, sticky="ew", padx=8, pady=5)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=12, pady=6)

        ttk.Button(
            control_frame,
            text="Start Auto",
            command=self.controller.start_auto,
        ).pack(side="left")

        ttk.Button(
            control_frame,
            text="Start Step Mode",
            command=self.controller.start_step_mode,
        ).pack(side="left", padx=8)

        ttk.Button(
            control_frame,
            text="Continue",
            command=self.controller.continue_simulation,
        ).pack(side="left", padx=8)

        ttk.Button(
            control_frame,
            text="Pause",
            command=self.controller.pause,
        ).pack(side="left", padx=8)

        ttk.Button(
            control_frame,
            text="Stop",
            command=self.controller.stop,
        ).pack(side="left", padx=8)

        ttk.Button(
            control_frame,
            text="Clear Log",
            command=self.clear_log,
        ).pack(side="left", padx=8)

        log_frame = ttk.LabelFrame(self.root, text="Event Log")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        self.log_text = tk.Text(
            log_frame,
            height=12,
            font=("Consolas", 9),
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)

    def _create_player_area(self, parent, player_index: int):
        frame = ttk.LabelFrame(parent, text=f"Player {player_index + 1}")

        header = ttk.Label(
            frame,
            text=self._player_header_text(player_index, "Waiting"),
            font=("Arial", 9, "bold"),
            justify="center",
        )
        header.pack(pady=(4, 1))

        bid_label = ttk.Label(
            frame,
            text="Bid: -",
            font=("Arial", 9),
            justify="center",
        )
        bid_label.pack(pady=(0, 3))

        card_frame = ttk.Frame(frame)
        card_frame.pack(padx=4, pady=4)

        self.player_header_labels[player_index] = header
        self.player_bid_labels[player_index] = bid_label
        self.player_card_frames[player_index] = card_frame

        return frame

    def _create_centre_area(self, parent):
        frame = ttk.LabelFrame(parent, text="Current Trick")

        for row in range(3):
            frame.rowconfigure(row, weight=1)

        for column in range(3):
            frame.columnconfigure(column, weight=1)

        self.current_trick_label = ttk.Label(
            frame,
            text="No trick yet",
            font=("Arial", 10, "bold"),
            justify="center",
            anchor="center",
        )
        self.current_trick_label.grid(
            row=1,
            column=1,
            padx=8,
            pady=8,
            sticky="nsew",
        )

        self.played_card_labels[0] = self._create_played_card_label(frame, "P1")
        self.played_card_labels[0].grid(row=0, column=1, padx=6, pady=4, sticky="nsew")

        self.played_card_labels[1] = self._create_played_card_label(frame, "P2")
        self.played_card_labels[1].grid(row=1, column=2, padx=6, pady=4, sticky="nsew")

        self.played_card_labels[2] = self._create_played_card_label(frame, "P3")
        self.played_card_labels[2].grid(row=2, column=1, padx=6, pady=4, sticky="nsew")

        self.played_card_labels[3] = self._create_played_card_label(frame, "P4")
        self.played_card_labels[3].grid(row=1, column=0, padx=6, pady=4, sticky="nsew")

        return frame

    def _create_played_card_label(self, parent, player_text: str):
        return ttk.Label(
            parent,
            text=f"{player_text}\n-",
            font=("Consolas", 11, "bold"),
            justify="center",
            anchor="center",
            relief="ridge",
            padding=(8, 5),
            width=8,
        )

    def _player_header_text(
        self,
        player_index: int,
        status: str,
        is_qabool: bool = False,
        is_first_shooter: bool = False,
    ) -> str:
        markers = []

        if is_qabool:
            markers.append("Sahib Al-Qabool")

        if is_first_shooter:
            markers.append("First Shooter")

        marker_text = ""
        if markers:
            marker_text = "\n" + " | ".join(markers)

        return f"Player {player_index + 1}{marker_text}\n{status}"

    def set_status(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.config(text=message)
    
    def set_shota_info(
        self,
        trump: str,
        qabool: str = "-",
        bid: str = "-",
        first_shooter: str = "-",
    ) -> None:
        if self.shota_info_label is not None:
            self.shota_info_label.config(
                text=(
                    f"Trump: {trump} | "
                    f"Qabool: {qabool} | "
                    f"Shota Bid: {bid} | "
                    f"First Shooter: {first_shooter}"
                )
            )

    def set_current_trick(self, message: str) -> None:
        if self.current_trick_label is not None:
            self.current_trick_label.config(text=message)

    def set_player_status(
        self,
        player_index: int,
        message: str,
        is_qabool: bool = False,
        is_first_shooter: bool = False,
    ) -> None:
        if 0 <= player_index < len(self.player_header_labels):
            label = self.player_header_labels[player_index]

            if label is not None:
                label.config(
                    text=self._player_header_text(
                        player_index,
                        message,
                        is_qabool=is_qabool,
                        is_first_shooter=is_first_shooter,
                    )
                )

                if is_qabool:
                    label.config(foreground="red")
                elif is_first_shooter:
                    label.config(foreground="green")
                else:
                    label.config(foreground="black")

    def set_player_bid(self, player_index: int, bid: str) -> None:
        if 0 <= player_index < len(self.player_bid_labels):
            label = self.player_bid_labels[player_index]
            if label is not None:
                label.config(text=f"Bid: {bid}")

    def set_player_hand(self, player_index: int, cards) -> None:
        """
        Show player's cards as compact card boxes.
        cards must be a list like ["A♠", "10♥", "K♦"].
        """

        if isinstance(cards, str):
            cards = [cards]

        if 0 <= player_index < len(self.player_card_frames):
            frame = self.player_card_frames[player_index]

            if frame is None:
                return

            for widget in frame.winfo_children():
                widget.destroy()

            if not cards:
                ttk.Label(
                    frame,
                    text="No cards",
                    font=("Arial", 9),
                ).grid(row=0, column=0)
                return

            for card_index, card_text in enumerate(cards):
                card_label = ttk.Label(
                    frame,
                    text=card_text,
                    font=("Consolas", 11, "bold"),
                    relief="ridge",
                    padding=(5, 3),
                    justify="center",
                    width=4,
                )

                row = card_index // 7
                column = card_index % 7

                card_label.grid(
                    row=row,
                    column=column,
                    padx=2,
                    pady=2,
                )

    def set_played_cards(self, played_cards) -> None:
        self.clear_played_cards()

        for player_id, card_text in played_cards:
            if player_id in self.played_card_labels:
                self.played_card_labels[player_id].config(
                    text=f"P{player_id + 1}\n{card_text}"
                )

    def clear_played_cards(self) -> None:
        for player_id, label in self.played_card_labels.items():
            label.config(text=f"P{player_id + 1}\n-")

    def reset_player_statuses(self) -> None:
        for index in range(4):
            self.set_player_status(index, "Waiting")
            self.set_player_bid(index, "-")
            self.set_player_hand(index, [])

        self.clear_played_cards()

    def log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = WistAILabApp()
    app.run()