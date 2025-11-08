import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import HomeIcon from '@mui/icons-material/Home';
import InfoIcon from '@mui/icons-material/Info';
import MicIcon from '@mui/icons-material/Mic';
import { BottomNavigation, BottomNavigationAction, Fab, Paper } from '@mui/material';
import { Slot, usePathname, useRouter, useSegments } from 'expo-router';
import React from 'react';

// Define enum for tab names
enum TabName {
  Home = 'home',
  Record = 'record',
  About = 'about',
}

export default function TabLayout() {
  const router = useRouter();
  const segments = useSegments(); // current route segments
  const colorScheme = useColorScheme();
  const themeColors = Colors[colorScheme ?? 'light'];
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
    <Paper sx={{ minHeight: '100%' }}>
      {/* Floating Record Button */}
      <Fab
        onClick={() => router.replace('/record')}
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
      <Paper
        sx={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: themeColors.background,
          color: themeColors.text,
        }}
        elevation={3}
      >
        <BottomNavigation value={currentTab} onChange={handleChange} showLabels>
          <BottomNavigationAction
            label="Home"
            value={TabName.Home}
            icon={<HomeIcon />}
            sx={{ color: currentTab === TabName.Home ? themeColors.tint : 'gray' }}
            onClick={() => router.replace('/')}
          />
          <BottomNavigationAction
            label="About"
            value={TabName.About}
            icon={<InfoIcon />}
            sx={{ color: currentTab === TabName.About ? themeColors.tint : 'gray' }}
            onClick={() => router.replace('/about')}
          />
        </BottomNavigation>
      </Paper>

      <Slot />
    </Paper>
  );
}
