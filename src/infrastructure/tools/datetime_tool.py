# datetime_tool.py

import re
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class DateTimeTool:
    name = "datetime"
    description = (
        "Donne la date et l'heure actuelles. Supporte les timezones IANA "
        "(ex: Europe/Paris, Africa/Ouagadougou) ou les noms de pays "
        "(ex: France, Burkina Faso). "
        "UTC par défaut."
    )
    args_schema = {
        "timezone": (
            "Optionnel. Timezone IANA (ex: Europe/Paris) ou pays (ex: France, Burkina)."
        ),
        "country": "Optionnel. Nom du pays pour obtenir l'heure locale.",
    }
    optional_args = ("timezone", "country")
    return_direct = True
    trigger_words: tuple[str, ...] = (
        "heure",
        "date",
        "aujourd'hui",
        "maintenant",
        "today",
        "time",
    )

    _country_to_timezone = {
        "burkina faso": "Africa/Ouagadougou",
        "burkina": "Africa/Ouagadougou",
        "france": "Europe/Paris",
        "côte d'ivoire": "Africa/Abidjan",
        "cote d'ivoire": "Africa/Abidjan",
        "mali": "Africa/Bamako",
        "sénégal": "Africa/Dakar",
        "senegal": "Africa/Dakar",
        "usa": "America/New_York",
        "états-unis": "America/New_York",
        "etats-unis": "America/New_York",
        "angleterre": "Europe/London",
        "uk": "Europe/London",
        "allemagne": "Europe/Berlin",
        "japon": "Asia/Tokyo",
        "canada": "America/Toronto",
        "belgique": "Europe/Brussels",
        "suisse": "Europe/Zurich",
        "italie": "Europe/Rome",
        "espagne": "Europe/Madrid",
        "portugal": "Europe/Lisbon",
        "brésil": "America/Sao_Paulo",
        "bresil": "America/Sao_Paulo",
        "australie": "Australia/Sydney",
        "inde": "Asia/Kolkata",
        "chine": "Asia/Shanghai",
        "russie": "Europe/Moscow",
        "maroc": "Africa/Casablanca",
        "algérie": "Africa/Algiers",
        "algerie": "Africa/Algiers",
        "tunisie": "Africa/Tunis",
        "nigeria": "Africa/Lagos",
        "afrique du sud": "Africa/Johannesburg",
        "egypte": "Africa/Cairo",
    }

    async def execute(self, **kwargs: Any) -> str:
        timezone_name = str(
            kwargs.get("timezone", "") or kwargs.get("timeZone", "") or ""
        ).strip()
        country = str(kwargs.get("country", "") or "").strip()

        # Si country est fourni, le convertir en timezone
        if country and not timezone_name:
            timezone_name = self._resolve_country(country) or ""

        # Si timezone fourni est un pays, le convertir
        if timezone_name and timezone_name.lower() in self._country_to_timezone:
            timezone_name = self._country_to_timezone[timezone_name.lower()]
        elif timezone_name and "/" not in timezone_name:
            # Essayer de résoudre comme pays
            resolved = self._resolve_country(timezone_name)
            if resolved:
                timezone_name = resolved

        # Fallback UTC
        if not timezone_name:
            timezone_name = "UTC"

        tz = self._resolve_timezone(timezone_name)

        if tz is None:
            pays_proches = self._suggest_countries(timezone_name)
            suggestion = ""
            if pays_proches:
                suggestion = f" Pays supportés: {', '.join(pays_proches[:5])}."
            return f"Timezone inconnue: {timezone_name}.{suggestion}"

        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")

    def _resolve_country(self, text: str) -> str | None:
        """Convertit un nom de pays en timezone IANA."""
        text_lower = text.lower().strip()
        for country, tz in self._country_to_timezone.items():
            if country in text_lower or text_lower in country:
                return tz
        return None

    def _suggest_countries(self, partial: str) -> list[str]:
        """Suggère des pays proches du texte fourni."""
        partial_lower = partial.lower()
        matches = [
            country
            for country in self._country_to_timezone
            if partial_lower in country or country in partial_lower
        ]
        return matches[:5]

    def _resolve_timezone(self, timezone_name: str) -> tzinfo | None:
        if timezone_name.upper() == "UTC":
            return UTC

        offset_tz = self._parse_utc_offset(timezone_name)
        if offset_tz is not None:
            return offset_tz

        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            return self._fallback_timezone(timezone_name)

    def _parse_utc_offset(self, timezone_name: str) -> timezone | None:
        normalized = timezone_name.strip().upper().replace(" ", "")
        match = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?", normalized)
        if not match:
            return None

        sign, hours_text, minutes_text = match.groups()
        hours = int(hours_text)
        minutes = int(minutes_text or "0")
        if hours > 14 or minutes > 59:
            return None

        delta = timedelta(hours=hours, minutes=minutes)
        if sign == "-":
            delta = -delta

        label = f"UTC{sign}{hours:02d}:{minutes:02d}"
        return timezone(delta, label)

    def _fallback_timezone(self, timezone_name: str) -> timezone | None:
        fallback_offsets = {
            "europe/paris": timezone(timedelta(hours=2), "UTC+02:00"),
            "africa/ouagadougou": timezone(timedelta(hours=0), "UTC+00:00"),
            "africa/abidjan": timezone(timedelta(hours=0), "UTC+00:00"),
            "africa/bamako": timezone(timedelta(hours=0), "UTC+00:00"),
            "africa/dakar": timezone(timedelta(hours=0), "UTC+00:00"),
            "africa/lagos": timezone(timedelta(hours=1), "UTC+01:00"),
            "africa/casablanca": timezone(timedelta(hours=1), "UTC+01:00"),
            "africa/algiers": timezone(timedelta(hours=1), "UTC+01:00"),
            "africa/tunis": timezone(timedelta(hours=1), "UTC+01:00"),
            "africa/cairo": timezone(timedelta(hours=3), "UTC+03:00"),
            "africa/johannesburg": timezone(timedelta(hours=2), "UTC+02:00"),
            "europe/london": timezone(timedelta(hours=1), "UTC+01:00"),
            "europe/berlin": timezone(timedelta(hours=2), "UTC+02:00"),
            "europe/moscow": timezone(timedelta(hours=3), "UTC+03:00"),
            "america/new_york": timezone(timedelta(hours=-4), "UTC-04:00"),
            "america/toronto": timezone(timedelta(hours=-4), "UTC-04:00"),
            "america/sao_paulo": timezone(timedelta(hours=-3), "UTC-03:00"),
            "asia/tokyo": timezone(timedelta(hours=9), "UTC+09:00"),
            "asia/kolkata": timezone(timedelta(hours=5, minutes=30), "UTC+05:30"),
            "asia/shanghai": timezone(timedelta(hours=8), "UTC+08:00"),
            "australia/sydney": timezone(timedelta(hours=11), "UTC+11:00"),
        }
        return fallback_offsets.get(timezone_name.lower())
