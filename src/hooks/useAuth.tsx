import React, { createContext, useContext, ReactNode } from 'react';

interface AuthContextType {
  user: any;
  login: (apiKey: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const login = (apiKey: string) => {
    localStorage.setItem('api_key', apiKey);
  };

  const logout = () => {
    localStorage.removeItem('api_key');
  };

  const value = {
    user: null,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 