import { Colors } from '@/constants/theme';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import MicIcon from '@mui/icons-material/Mic';
import { BottomNavigation, BottomNavigationAction, Box, Fab } from '@mui/material';
import { Slot, usePathname, useRouter, useSegments } from 'expo-router';
import React from 'react';

// Define enum for tab names
enum TabName {
  Home = 'home',
  Record = 'record',
  About = 'about',
}

export default function TabLayout() {
  const themeColors = Colors.dark;

  const router = useRouter();
  const segments = useSegments(); // current route segments
  const pathname = usePathname(); // e.g., '/', '/about', '/record'

  const currentTab: TabName = (() => {
    switch (pathname) {
      case '/about':
        return TabName.About;
      case '/record':
        return TabName.Record;
      case '/':
        return TabName.Home;
      default:
        return TabName.Home;
    }
  })();

  const handleChange = (_event: React.SyntheticEvent, newValue: TabName) => {
    router.replace(`/${newValue === TabName.Home ? '' : newValue}`);
    console.log(currentTab);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: '#DDD6F3',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
        }}
      >
        <Slot />
      </Box>

      {/* Floating Record Button */}
      <Fab
        // onClick={() => router.replace('/record')}
        style={{
          position: 'fixed',
          bottom: 40,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 1,
          backgroundColor: themeColors.tint,
          color: themeColors.background,
        }}
      >
        <MicIcon />
      </Fab>

      {/* Bottom Navigation */}
      <BottomNavigation
        value={currentTab}
        onChange={handleChange}
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          color: themeColors.text,
        }}
      >
        <BottomNavigationAction
          // label="Home"
          value={TabName.Home}
          icon={<HomeIcon />}
          sx={{ color: currentTab === TabName.Home ? themeColors.tint : 'gray' }}
          onClick={() => router.replace('/')}
        />
        <BottomNavigationAction
          // label="About"
          value={TabName.About}
          icon={<InfoIcon />}
          sx={{ color: currentTab === TabName.About ? themeColors.tint : 'gray' }}
          onClick={() => router.replace('/about')}
        />
      </BottomNavigation>
    </Box>
  );
}
