import 'dart:convert';
import 'package:flutter1/model/weather.dart';
import 'package:http/http.dart' as http;


class WeatherService {



  Future<Weather> getWeatherData(String place) async {
    try {
      final queryParameters = {
        'key': 'a0c4b603719941e4a1803122231403',
        'q': place,
      };
      final uri = Uri.http('api.weatherapi.com', '/v1/current.json', queryParameters);
      final response = await http.get(uri);
      if(response.statusCode == 200) {
        return Weather.fromJson(jsonDecode(response.body));
      } else {
        throw Exception("Can not get weather");
      }
    } catch(e) {
      rethrow;
    }
  }
}