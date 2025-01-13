import 'package:firebase_core/firebase_core.dart';
import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter/material.dart';
import 'package:flutter1/Lab.dart';
import 'package:flutter1/Setting.dart';
import 'package:flutter1/recommendation.dart';
import 'model/weather.dart';
import 'package:flutter1/services/weather_service.dart';
import 'package:flutter1/landingPage/landingPage.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:get/get.dart';
import 'package:flutter1/LocaleString.dart';
import 'package:shared_preferences/shared_preferences.dart';


const String apiUrl = 'https://fluttermodeldeploy.azurewebsites.net';

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(
     GetMaterialApp(
      debugShowCheckedModeBanner: false,
      translations: LocalString(),
      locale: Locale('en','US'),
      title: 'Agriguru',
      home: MyApp(),
    ),
  );
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  HomePageState createState() => HomePageState();
}

class HomePageState extends State<HomePage> {

  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  var currentIndex = 0;
  double _nValue = 0.0;
  double _pValue = 5.0;
  double _kValue = 5.0;

  WeatherService weatherService = WeatherService();
  Weather weather = Weather();

  String currentWeather = "";
  double tempC = 0;
  double rain= 0;
  int humid = 0;
  String name = "";
  String country = "";
  String region = "";
  String icon="";
  String windDirection="";
  String localTime="";

  @override
  void initState() {
    super.initState();
    getWeather();
  }
  void getWeather() async {

    dynamic result = await Geolocator.requestPermission();
    if (result == LocationPermission.denied) {
      Fluttertoast.showToast(
          msg: "Permission denied",
          toastLength: Toast.LENGTH_SHORT,
          gravity: ToastGravity.CENTER,
          timeInSecForIosWeb: 1,
          backgroundColor: Colors.red,
          textColor: Colors.white,
          fontSize: 16.0
      );
      } else if (result == LocationPermission.deniedForever) {
      Fluttertoast.showToast(
          msg: "Permission denied forever",
          toastLength: Toast.LENGTH_SHORT,
          gravity: ToastGravity.CENTER,
          timeInSecForIosWeb: 1,
          backgroundColor: Colors.red,
          textColor: Colors.white,
          fontSize: 16.0
      );
      } else {
        Fluttertoast.showToast(
            msg: "Permission granted",
            toastLength: Toast.LENGTH_SHORT,
            gravity: ToastGravity.CENTER,
            timeInSecForIosWeb: 1,
            backgroundColor: Colors.red,
            textColor: Colors.white,
            fontSize: 16.0
        );
      }
    Position position = await Geolocator.getCurrentPosition(desiredAccuracy: LocationAccuracy.high);
    double latitude = position.latitude;
    double longitude = position.longitude;
    String pl="$latitude, $longitude";
    print("lol is : $pl");
    weather = await weatherService.getWeatherData(pl);

    setState(() {
      currentWeather = weather.condition;
      tempC = weather.temperatureC;
      rain = weather.rain;
      humid = weather.humidity;
      name = weather.name;
      country = weather.country;
      region = weather.region;
      icon = "http:${weather.icon}";
      windDirection= weather.windDirection;
      localTime=weather.localTime;
    });
    print(weather.temperatureC);
    print(weather.rain);
    print(weather.condition);
    print(weather.humidity);
    print(weather.name);
    print(weather.country);
    print(weather.region);
    print(weather.icon);
    print(weather.windDirection);
    print(weather.localTime);
    localTime=localTime.substring(0,10);
    print("localtime is : $localTime");
  }

  @override
  Widget build(BuildContext context) {
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: displayHeight * 0.125,
        title: Center(
          child: Padding(
            padding: EdgeInsets.all(displayWidth * 0.01),
            child: Row(
              children: [
                Image.asset('assets/images/logo.png',
                    width: displayWidth*0.3,
                ),
                SizedBox(
                  width: displayWidth*0.425,
                ),
                TextButton(
                  onPressed: (){},
                  child: const CircleAvatar(
                    backgroundImage: AssetImage('assets/images/avatar.jpg'),
                    radius: 25,
                  ),
                )],
            ),
          ),
        ),
        backgroundColor: const Color(0xFFFBFCFA),
        elevation: 0,
        automaticallyImplyLeading: false,
      ),

      body: SingleChildScrollView(
        child: Center(
          child: Column(
            children: [

              Container(
                margin: const EdgeInsets.fromLTRB(20, 0, 20, 10),
                width: displayWidth * 1,
                height: displayWidth * 0.575,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: const Color(0xFFECEEE6),
                ),
                child: Column(
                  children: [
                    Container(
                      margin: EdgeInsets.all(displayWidth * .015),
                      width: displayWidth * 0.8,
                      height: displayWidth * 0.125,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child:Row(
                        children: [
                          Text(currentWeather,
                            style: const TextStyle(fontWeight: FontWeight.w500,
                              fontSize: 22.5,
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.375,
                          ),
                          Row(
                            children: [
                              Image.network(icon),
                              Text(tempC.toString(),
                                style: const TextStyle(fontWeight: FontWeight.w400,
                                  fontSize: 25,
                                  color: Color(0xFFE59A54),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      height: displayWidth * 0.05,
                    ),
                    Container(
                      margin: EdgeInsets.fromLTRB(displayWidth*0.075,0,0,0),
                      width: displayWidth * 0.8,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child:Row(
                        children: [
                          Center(
                            child: Column(
                              children: [
                                const Icon(Icons.thermostat,
                                  color: Color(0xFF1C9C73),
                                  size: 35.0,
                                ),
                                Text(tempC.toString(),
                                  style: const TextStyle(color: Color(0xFF1C9C73)),),
                                Text('Dash_temp'.tr,
                                  style: TextStyle(color: Color(0xFF1C9C73)),),
                              ],
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.075,
                          ),
                          Center(
                            child: Column(
                              children: [
                                const Icon(
                                  Icons.water_drop_outlined,
                                  color: Color(0xFF1C9C73),
                                  size: 35.0,
                                ),
                                Text('$humid %',
                                  style: const TextStyle(color: Color(0xFF1C9C73)),),
                                Text('Dash_humi'.tr,
                                  style: TextStyle(color: Color(0xFF1C9C73)),),
                              ],
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.075,
                          ),
                          Center(
                            child: Column(
                              children: [
                                const Icon(
                                  Icons.cloud,
                                  color: Color(0xFF1C9C73),
                                  size: 35.0,
                                ),
                                Text('$rain mm',
                                  style: const TextStyle(color: Color(0xFF1C9C73)),),
                                Text('Dash_rain'.tr,
                                  style: TextStyle(color: Color(0xFF1C9C73)),),
                              ],
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.075,
                          ),
                          Center(
                            child: Column(
                              children:[
                                const Icon(
                                  Icons.waves_rounded,
                                  color: Color(0xFF1C9C73),
                                  size: 35.0,
                                ),
                                Text(windDirection,
                                  style: const TextStyle(color: Color(0xFF1C9C73)),),
                                Text('Dash_direc'.tr,
                                  style: TextStyle(color: Color(0xFF1C9C73)),),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      height: displayWidth * 0.05,
                    ),
                    Container(
                      margin: EdgeInsets.all(displayWidth * .015),
                      width: displayWidth * 0.8,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child: Row(
                        children: [
                          Container(
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(12),
                              color: const Color(0xFFECEEE6),
                            ),
                            child: Column(
                              children: [
                                Text('$name ,',
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w500,
                                  ),
                                  textAlign: TextAlign.left,
                                ),
                                Text(region,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w500,
                                  ),
                                  textAlign: TextAlign.left,
                                ),
                              ],
                            ),
                          ),
                          SizedBox(
                            width: displayWidth*0.35,
                          ),
                          Container(
                            padding: const EdgeInsets.all(7.5),
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(12),
                              color: const Color(0xFFD9D9D9),
                            ),
                            child: Center(
                              child: Row(
                                children: [
                                  Text(localTime+" "),
                                  Image.asset('assets/images/calender.png',
                                    width: displayWidth*0.065,),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),


              ),

              Container(
                margin: EdgeInsets.fromLTRB(20, 10, 20, displayHeight*0.15),
                width: displayWidth * 1,
                height: displayHeight * 0.4,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: const Color(0xFFECEEE6),
                ),
                child: Column(
                  children: [
                    Container(
                      padding:EdgeInsets.all( displayWidth * 0.05 ),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child: Row(
                        children: [
                           Text(
                            'Soil_det'.tr,
                            style: TextStyle(
                                fontWeight: FontWeight.w500,
                                fontSize: 25
                            ),
                          ),
                          SizedBox(
                            width: displayWidth*0.2,
                          ),
                          ElevatedButton(onPressed: (){

                            Future<void> getCropPredictions(double n, double p, double k, double temp, double humid, double ph, double rain) async {
                              try {
                                final response = await http.get(Uri.parse("$apiUrl?n=$n&p=$p&k=$k&temp=$temp&humid=$humid&ph=$ph&rain=$rain"));
                                if (response.statusCode == 200) {
                                  Map<String, dynamic> data = json.decode(response.body);
                                  print("worked");
                                  print("crops: $data");
                                  print(data["crop1"]);
                                  print(data["crop2"]);
                                  print(data["crop3"]);

                                  SharedPreferences prefs = await SharedPreferences.getInstance();
                                  prefs.setDouble('n', n);
                                  prefs.setDouble('p', p);
                                  prefs.setDouble('k', k);
                                  prefs.setDouble('temp', temp);
                                  prefs.setDouble('humid', humid);
                                  prefs.setDouble('ph', ph);
                                  prefs.setDouble('rain', rain);
                                  prefs.setString('crop1', data['crop1']);
                                  prefs.setString('crop2', data['crop2']);
                                  prefs.setString('crop3', data['crop3']);

                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(builder: (context) => Tabs(),
                                      settings: RouteSettings(
                                      arguments: data,
                                    ),),
                                  );
                                } else {
                                  throw Exception('Failed to load crop predictions');
                                }
                              } catch (e) {
                                dynamic l="$apiUrl?n=$n&p=$p&k=$k&temp=$temp&humid=$humid&ph=$ph&rain=$rain";
                                throw Exception('Failed to connect to server: $e with url $l');
                              }
                            }


                            String inputText = _controller.text;
                            double ph=double.parse(inputText);
                            double humidity=humid.toDouble();
                            print("ph is : $ph");
                            print("N is : $_nValue");
                            print("P is : $_pValue");
                            print("K is : $_kValue");
                            print("temp is : $tempC");
                            print("humid is : $humidity");
                            print("rain is : $rain");
                            getCropPredictions(_nValue,_pValue,_kValue,tempC,humidity,ph,rain);
                          },
                            style: ButtonStyle(
                              backgroundColor: MaterialStateProperty.all<Color>(const Color(0xFF1C9C73)),
                              shape: MaterialStateProperty.all<RoundedRectangleBorder>(RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              ),
                            ),
                            child: Row(
                              children: [
                                Text('Predict'.tr),
                                SizedBox(
                                  width: displayWidth*0.01,
                                ),
                                const Icon(Icons.play_circle_fill_rounded),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.fromLTRB(45, 5, 5, 5),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xffeceee6),
                      ),
                      child: Row(
                        children: [
                          Text('ph_is'.tr,
                          style: TextStyle(
                            fontSize: 15,
                            color: Color(0xff696a63)
                          ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.05,
                          ),
                          SizedBox(
                            width: displayWidth * 0.5,
                            height: displayHeight * 0.05,
                            child: TextFormField(
                              controller: _controller,
                              decoration: const InputDecoration(
                                focusedBorder: OutlineInputBorder(
                                  borderRadius: BorderRadius.all(
                                    Radius.circular(5.0),
                                  ),
                                    borderSide: BorderSide(color: Color(0xFF4FAA8D)),
                                ),
                                filled: true,
                                fillColor: Colors.white,
                                enabledBorder: UnderlineInputBorder(

                                  borderSide: BorderSide(color: Color(0xFF4FAA8D),
                                  ),

                              ),
                            ),

                          ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                     height: displayHeight * 0.01,
                    ),
                    Container(
                      padding:const EdgeInsets.fromLTRB(45, 5, 0, 5),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child:Row(
                        children: [
                          Text('N_is'.tr,
                            style: TextStyle(
                              fontSize: 15,
                              color: Color(0xFF696A63),
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.05,
                          ),
                          SizedBox(
                            width: displayWidth * 0.575,
                            child: SliderTheme(

                              data: SliderTheme.of(context).copyWith(
                                showValueIndicator: ShowValueIndicator.always,
                                inactiveTrackColor: const Color(0XFFFBFCFA),
                                activeTrackColor: const Color(0xFF4FAA8D),
                              ),

                              child:  Slider(
                                value: _nValue,
                                min: 0,
                                max: 140,
                                divisions: 140,
                                label: _nValue.round().toString(),
                                onChanged: (double value) {
                                  setState(() {
                                    _nValue = value;
                                  });
                                },
                                thumbColor: const Color(0xFFE59A54),

                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding:const EdgeInsets.fromLTRB(45, 5, 0, 5),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child:Row(
                        children: [
                          Text('P_is'.tr,
                            style: TextStyle(
                              fontSize: 15,
                              color: Color(0xFF696A63),
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.05,
                          ),
                          SizedBox(
                            width: displayWidth * 0.575,
                            child: SliderTheme(

                              data: SliderTheme.of(context).copyWith(
                                showValueIndicator: ShowValueIndicator.always,
                                inactiveTrackColor: const Color(0XFFFBFCFA),
                                activeTrackColor: const Color(0xFF9DD5DF),

                              ),
                              child:  Slider(
                                value: _pValue,
                                min: 5,
                                max: 145,
                                divisions: 140,
                                label: _pValue.round().toString(),
                                onChanged: (double value) {
                                  setState(() {
                                    _pValue = value;
                                  });
                                },
                                thumbColor: const Color(0xFFE59A54),

                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding:const EdgeInsets.fromLTRB(45, 5, 0, 5),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: const Color(0xFFECEEE6),
                      ),
                      child:Row(
                        children: [
                          Text('K_is'.tr,
                            style: TextStyle(
                              fontSize: 15,
                              color: Color(0xFF696A63),
                            ),
                          ),
                          SizedBox(
                            width: displayWidth * 0.05,
                          ),
                          SizedBox(
                            width: displayWidth * 0.575,
                            child: SliderTheme(

                              data: SliderTheme.of(context).copyWith(
                                showValueIndicator: ShowValueIndicator.always,
                                inactiveTrackColor: const Color(0XFFFBFCFA),
                                activeTrackColor: const Color(0xFFEBB8AF),

                              ),

                              child:  Slider(
                                value: _kValue,
                                min: 5,
                                max: 205,
                                divisions: 200,
                                label: _kValue.round().toString(),
                                onChanged: (double value) {
                                  setState(() {
                                    _kValue = value;
                                  });
                                },
                                thumbColor: const Color(0xFFE59A54),

                              ),
                            ),
                          ),

                        ],
                      ),
                    ),
                  ],
                ),
              ),

            ],
          ),
        ),
      ),

      extendBody: true,
      bottomNavigationBar: FloatingNavigationBar(
        backgroundColor: const Color(0xffD9D9D9),
        iconColor: Colors.black,
        barHeight: 70,
        textStyle: const TextStyle(
          color: Colors.black,
          fontSize: 14.0,
        ),
        indicatorColor: const Color(0xff1C9C73),
        indicatorWidth: 30,
        iconSize: 25.0,
        items: [
          NavBarItems(icon: Icons.home,title: "Nav_home".tr),
          NavBarItems(icon: Icons.addchart,title: "Nav_crop".tr),
          NavBarItems(icon: Icons.location_on, title: "Nav_lab".tr),
          NavBarItems(icon: Icons.settings_rounded,title: "Nav_setting".tr),
        ],
          onChanged: (value) {
          setState(() {
            currentIndex = value;
          });
          if(value==2){
            Navigator.push(context,
            MaterialPageRoute(builder: (context)=>const Labpage())
            );
          }
          if(value==3){
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) =>  SettingsPage()),
            );
          }
          }
      ),
    );
  }
}