"""Telegram bot interface adapter.

Uses python-telegram-bot v21+ async API. Polls long-poll by default. Handles
text messages, voice notes, and /start command. Exposes message stream to the
dispatcher via a single handler callback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.interfaces import (
    IncomingMessage,
    InterfaceProvider,
    OutgoingMessage,
)
from core.interfaces import (
    MessageHandler as CoreMessageHandler,
)
from core.registry import interface_registry

logger = logging.getLogger(__name__)


@interface_registry.register("telegram")
class TelegramInterface(InterfaceProvider):
    def __init__(
        self,
        *,
        token: str | None = None,
        allowed_user_ids: list[int] | None = None,
        **_kwargs: Any,
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not self.token:
            raise ValueError(
                "Telegram token not provided. Set TELEGRAM_BOT_TOKEN env or "
                "pass token in persona.interface.options."
            )
        self.allowed_user_ids = set(allowed_user_ids or [])
        self._app: Application | None = None
        self._handler: CoreMessageHandler | None = None

    async def start(self, handler: CoreMessageHandler) -> None:
        self._handler = handler
        self._app = ApplicationBuilder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self._app.add_handler(MessageHandler(filters.VOICE, self._on_voice))

        await self._app.initialize()
        await self._app.start()
        assert self._app.updater is not None
        await self._app.updater.start_polling()
        logger.info("Telegram bot started; polling.")

        try:
            await asyncio.Event().wait()
        finally:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    async def send(self, message: OutgoingMessage) -> None:
        if not self._app:
            raise RuntimeError("Interface not started; call start() first.")
        chat_id = int(message.user_id)
        if message.text:
            await self._app.bot.send_message(chat_id=chat_id, text=message.text)
        if message.audio:
            await self._app.bot.send_voice(chat_id=chat_id, voice=message.audio)

    async def _on_start(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(
                "Cześć! Jestem twoim asystentem głosowym."
            )

    async def _on_text(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._handler or not update.message or not update.effective_user:
            return
        if not self._is_allowed(update.effective_user.id):
            return
        msg = IncomingMessage(
            user_id=str(update.effective_chat.id) if update.effective_chat else "",
            text=update.message.text or "",
            metadata={"telegram_user_id": update.effective_user.id},
        )
        result = await self._handler(msg)
        if result:
            await self._reply(update, result)

    async def _on_voice(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        if (
            not self._handler
            or not update.message
            or not update.message.voice
            or not update.effective_user
        ):
            return
        if not self._is_allowed(update.effective_user.id):
            return
        voice_file = await update.message.voice.get_file()
        audio_bytes = bytes(await voice_file.download_as_bytearray())
        msg = IncomingMessage(
            user_id=str(update.effective_chat.id) if update.effective_chat else "",
            audio=audio_bytes,
            metadata={
                "telegram_user_id": update.effective_user.id,
                "duration": update.message.voice.duration,
            },
        )
        result = await self._handler(msg)
        if result:
            await self._reply(update, result)

    async def _reply(self, update: Update, msg: OutgoingMessage) -> None:
        if not update.message:
            return
        if msg.audio:
            await update.message.reply_voice(voice=msg.audio)
        if msg.text:
            await update.message.reply_text(msg.text)

    def _is_allowed(self, user_id: int) -> bool:
        if not self.allowed_user_ids:
            return True
        return user_id in self.allowed_user_ids
