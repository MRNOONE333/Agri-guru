import 'package:flutter/material.dart';
import 'package:flutter1/landingPage/pages/GetStartedPage.dart';
import 'package:get/get.dart';
import 'package:flutter1/LocaleString.dart  ';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // translations: LocalString(),
      // locale: Locale('en','US'),
      home: GetStartedPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}
