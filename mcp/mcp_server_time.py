#!/usr/bin/env python3
"""
Time MCP Server

This module implements a Model Context Protocol (MCP) server for time-related operations.
It provides tools for getting current time, converting between timezones, and performing
date calculations.
"""

from fastmcp import FastMCP
from datetime import datetime, timedelta
import pytz
import argparse
import logging
import sys

# Initialize FastMCP server
mcp = FastMCP("time", log_level="INFO")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Time MCP Server")
parser.add_argument("--local-timezone", default="UTC", help="Local timezone (default: UTC)")
args = parser.parse_args()

# Set local timezone
try:
    LOCAL_TIMEZONE = pytz.timezone(args.local_timezone)
    logging.info(f"Local timezone set to: {args.local_timezone}")
except pytz.exceptions.UnknownTimeZoneError:
    logging.error(f"Unknown timezone: {args.local_timezone}")
    logging.error("Defaulting to UTC")
    LOCAL_TIMEZONE = pytz.UTC

@mcp.tool()
async def get_current_time(timezone: str = None) -> str:
    """Get the current time in the specified timezone or local timezone if not specified.

    Args:
        timezone: The timezone to get the current time in (e.g., 'UTC', 'America/New_York')
                 If not provided, uses the local timezone set when starting the server.

    Returns:
        str: The current time in the specified timezone
    """
    try:
        # Get current time in UTC
        now_utc = datetime.now(pytz.UTC)

        # Convert to specified timezone or use local timezone
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                now = now_utc.astimezone(tz)
                return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone}'. Please use a valid timezone name like 'UTC' or 'America/New_York'."
        else:
            # Use local timezone
            now = now_utc.astimezone(LOCAL_TIMEZONE)
            return now.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        return f"Error getting current time: {str(e)}"

@mcp.tool()
async def convert_timezone(time_str: str, from_timezone: str, to_timezone: str) -> str:
    """Convert a time from one timezone to another.

    Args:
        time_str: The time string to convert (format: 'YYYY-MM-DD HH:MM:SS')
        from_timezone: The source timezone (e.g., 'UTC', 'America/New_York')
        to_timezone: The target timezone (e.g., 'UTC', 'America/New_York')

    Returns:
        str: The converted time in the target timezone
    """
    try:
        # Parse the input time
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"Error: Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS' format."

        # Set source timezone
        try:
            from_tz = pytz.timezone(from_timezone)
            dt = from_tz.localize(dt)
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Error: Unknown source timezone '{from_timezone}'."

        # Convert to target timezone
        try:
            to_tz = pytz.timezone(to_timezone)
            converted_dt = dt.astimezone(to_tz)
            return converted_dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Error: Unknown target timezone '{to_timezone}'."
    except Exception as e:
        return f"Error converting timezone: {str(e)}"

@mcp.tool()
async def get_timezone_difference(timezone1: str, timezone2: str) -> str:
    """Get the time difference between two timezones.

    Args:
        timezone1: The first timezone (e.g., 'UTC', 'America/New_York')
        timezone2: The second timezone (e.g., 'UTC', 'America/New_York')

    Returns:
        str: The time difference between the two timezones
    """
    try:
        # Get current time in UTC
        now_utc = datetime.now(pytz.UTC)

        # Convert to first timezone
        try:
            tz1 = pytz.timezone(timezone1)
            time1 = now_utc.astimezone(tz1)
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Error: Unknown timezone '{timezone1}'."

        # Convert to second timezone
        try:
            tz2 = pytz.timezone(timezone2)
            time2 = now_utc.astimezone(tz2)
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Error: Unknown timezone '{timezone2}'."

        # Calculate offset difference in hours
        offset1 = time1.utcoffset().total_seconds() / 3600
        offset2 = time2.utcoffset().total_seconds() / 3600
        diff = offset1 - offset2

        # Format the result
        if diff == 0:
            return f"{timezone1} and {timezone2} are in the same timezone (no time difference)."
        elif diff > 0:
            return f"{timezone1} is {abs(diff)} hours ahead of {timezone2}."
        else:
            return f"{timezone1} is {abs(diff)} hours behind {timezone2}."
    except Exception as e:
        return f"Error getting timezone difference: {str(e)}"

@mcp.tool()
async def format_time(time_str: str = None, format_str: str = "%Y-%m-%d %H:%M:%S", timezone: str = None) -> str:
    """Format a time string or the current time using the specified format.

    Args:
        time_str: The time string to format (format: 'YYYY-MM-DD HH:MM:SS')
                 If not provided, uses the current time.
        format_str: The format string to use (default: '%Y-%m-%d %H:%M:%S')
                   See Python's datetime.strftime documentation for format codes.
        timezone: The timezone to use (e.g., 'UTC', 'America/New_York')
                 If not provided, uses the local timezone.

    Returns:
        str: The formatted time
    """
    try:
        if time_str:
            # Parse the input time
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

                # Set timezone if provided
                if timezone:
                    try:
                        tz = pytz.timezone(timezone)
                        dt = dt.replace(tzinfo=pytz.UTC).astimezone(tz)
                    except pytz.exceptions.UnknownTimeZoneError:
                        return f"Error: Unknown timezone '{timezone}'."
                else:
                    # Use local timezone
                    dt = dt.replace(tzinfo=pytz.UTC).astimezone(LOCAL_TIMEZONE)
            except ValueError:
                return f"Error: Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS' format."
        else:
            # Use current time
            dt = datetime.now(pytz.UTC)

            # Set timezone if provided
            if timezone:
                try:
                    tz = pytz.timezone(timezone)
                    dt = dt.astimezone(tz)
                except pytz.exceptions.UnknownTimeZoneError:
                    return f"Error: Unknown timezone '{timezone}'."
            else:
                # Use local timezone
                dt = dt.astimezone(LOCAL_TIMEZONE)

        # Format the time
        return dt.strftime(format_str)
    except Exception as e:
        return f"Error formatting time: {str(e)}"

@mcp.tool()
async def add_time(time_str: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0, timezone: str = None) -> str:
    """Add time to a given time string or the current time.

    Args:
        time_str: The time string to add to (format: 'YYYY-MM-DD HH:MM:SS')
        days: Number of days to add (default: 0)
        hours: Number of hours to add (default: 0)
        minutes: Number of minutes to add (default: 0)
        seconds: Number of seconds to add (default: 0)
        timezone: The timezone to use (e.g., 'UTC', 'America/New_York')
                 If not provided, uses the local timezone.

    Returns:
        str: The resulting time after addition
    """
    try:
        # Parse the input time
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"Error: Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS' format."

        # Set timezone if provided
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                dt = tz.localize(dt)
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone}'."
        else:
            # Use local timezone
            dt = LOCAL_TIMEZONE.localize(dt)

        # Add time
        dt = dt + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

        # Return the result
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        return f"Error adding time: {str(e)}"

@mcp.tool()
async def subtract_time(time_str: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0, timezone: str = None) -> str:
    """Subtract time from a given time string or the current time.

    Args:
        time_str: The time string to subtract from (format: 'YYYY-MM-DD HH:MM:SS')
        days: Number of days to subtract (default: 0)
        hours: Number of hours to subtract (default: 0)
        minutes: Number of minutes to subtract (default: 0)
        seconds: Number of seconds to subtract (default: 0)
        timezone: The timezone to use (e.g., 'UTC', 'America/New_York')
                 If not provided, uses the local timezone.

    Returns:
        str: The resulting time after subtraction
    """
    try:
        # Parse the input time
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"Error: Invalid time format. Please use 'YYYY-MM-DD HH:MM:SS' format."

        # Set timezone if provided
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                dt = tz.localize(dt)
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone}'."
        else:
            # Use local timezone
            dt = LOCAL_TIMEZONE.localize(dt)

        # Subtract time
        dt = dt - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

        # Return the result
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    except Exception as e:
        return f"Error subtracting time: {str(e)}"

@mcp.tool()
async def get_time_difference(time_str1: str, time_str2: str, timezone1: str = None, timezone2: str = None) -> str:
    """Get the time difference between two time strings.

    Args:
        time_str1: The first time string (format: 'YYYY-MM-DD HH:MM:SS')
        time_str2: The second time string (format: 'YYYY-MM-DD HH:MM:SS')
        timezone1: The timezone for the first time (e.g., 'UTC', 'America/New_York')
                  If not provided, uses the local timezone.
        timezone2: The timezone for the second time (e.g., 'UTC', 'America/New_York')
                  If not provided, uses the same timezone as timezone1.

    Returns:
        str: The time difference between the two times
    """
    try:
        # Parse the first time
        try:
            dt1 = datetime.strptime(time_str1, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"Error: Invalid format for first time. Please use 'YYYY-MM-DD HH:MM:SS' format."

        # Parse the second time
        try:
            dt2 = datetime.strptime(time_str2, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return f"Error: Invalid format for second time. Please use 'YYYY-MM-DD HH:MM:SS' format."

        # Set timezone for first time
        if timezone1:
            try:
                tz1 = pytz.timezone(timezone1)
                dt1 = tz1.localize(dt1)
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone1}'."
        else:
            # Use local timezone
            dt1 = LOCAL_TIMEZONE.localize(dt1)

        # Set timezone for second time
        if timezone2:
            try:
                tz2 = pytz.timezone(timezone2)
                dt2 = tz2.localize(dt2)
            except pytz.exceptions.UnknownTimeZoneError:
                return f"Error: Unknown timezone '{timezone2}'."
        elif timezone1:
            # Use same timezone as first time
            tz1 = pytz.timezone(timezone1)
            dt2 = tz1.localize(dt2)
        else:
            # Use local timezone
            dt2 = LOCAL_TIMEZONE.localize(dt2)

        # Calculate time difference
        diff = dt1 - dt2

        # Get total seconds
        total_seconds = abs(diff.total_seconds())

        # Calculate days, hours, minutes, seconds
        days = int(total_seconds // (24 * 3600))
        total_seconds %= (24 * 3600)
        hours = int(total_seconds // 3600)
        total_seconds %= 3600
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)

        # Format the result
        result = []
        if days > 0:
            result.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            result.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            result.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not result:
            result.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        if dt1 > dt2:
            return f"Time difference: {', '.join(result)} (first time is later)"
        elif dt2 > dt1:
            return f"Time difference: {', '.join(result)} (second time is later)"
        else:
            return "The times are identical (no difference)"
    except Exception as e:
        return f"Error getting time difference: {str(e)}"

@mcp.tool()
async def list_timezones() -> str:
    """List all available timezones.

    Returns:
        str: A list of all available timezones
    """
    try:
        # Get all timezones
        all_timezones = pytz.all_timezones

        # Group timezones by region
        regions = {}
        for tz in all_timezones:
            parts = tz.split('/')
            if len(parts) > 1:
                region = parts[0]
                if region not in regions:
                    regions[region] = []
                regions[region].append(tz)
            else:
                if 'Other' not in regions:
                    regions['Other'] = []
                regions['Other'].append(tz)

        # Format the result
        result = "Available Timezones:\n\n"
        for region, timezones in sorted(regions.items()):
            result += f"{region}:\n"
            for tz in sorted(timezones):
                result += f"  - {tz}\n"
            result += "\n"

        return result
    except Exception as e:
        return f"Error listing timezones: {str(e)}"

if __name__ == "__main__":
    logging.info("Starting Time MCP Server")
    logging.info(f"Local timezone: {LOCAL_TIMEZONE}")

    # Run MCP server
    mcp.run(transport='stdio')
