import 'dart:ffi';

class Weather {
  final double temperatureC;
  final String condition;
  final int humidity;
  final String name;
  final String region;
  final String country;
  final String icon;
  final double rain;
  final String windDirection;
  final String localTime;

  Weather({
    this.temperatureC = 0,
    this.rain = 0,
    this.condition = "Sunny",
    this.humidity = 0,
    this.name = "",
    this.region = "",
    this.country = "",
    this.icon = "",
    this.windDirection="",
    this.localTime="",
  });

  factory Weather.fromJson(Map<String, dynamic> json) {
    return Weather(
      temperatureC: json['current']['temp_c'],
      condition: json['current']['condition']['text'],
      icon: json['current']['condition']['icon'],
      humidity: json['current']['humidity'],
      name: json['location']['name'],
      region: json['location']['region'],
      country: json['location']['country'],
      rain: json['current']['precip_mm'],
      windDirection: json['current']['wind_dir'],
      localTime: json['location']['localtime'],
    );
  }
}